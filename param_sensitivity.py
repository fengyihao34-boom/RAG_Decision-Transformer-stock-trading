import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import chromadb
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ===== 加载数据 =====
df = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
stock = df[df['Symbol'] == 'GOOG'].copy().reset_index(drop=True)

stock['MA5']  = stock['Close'].rolling(5).mean()
stock['MA20'] = stock['Close'].rolling(20).mean()
delta         = stock['Close'].diff()
gain          = delta.where(delta > 0, 0).rolling(14).mean()
loss_         = (-delta.where(delta < 0, 0)).rolling(14).mean()
stock['RSI']  = 100 - (100 / (1 + gain / loss_))
ema12         = stock['Close'].ewm(span=12).mean()
ema26         = stock['Close'].ewm(span=26).mean()
stock['MACD'] = ema12 - ema26
stock         = stock.dropna().reset_index(drop=True)

train_stock = stock[stock['Date'] < '2022-01-01'].reset_index(drop=True)
test_stock  = stock[stock['Date'] >= '2022-01-01'].reset_index(drop=True)

feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']
scaler       = MinMaxScaler()
train_features = scaler.fit_transform(train_stock[feature_cols].values)
test_features  = scaler.transform(test_stock[feature_cols].values)

# 加载情感数据
df_sent = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_sentiment.csv')
df_sent['Date'] = pd.to_datetime(df_sent['Date'])
test_stock = test_stock.merge(df_sent, on='Date', how='left')
test_stock[['sent_pos','sent_neg','sent_neu']] = \
    test_stock[['sent_pos','sent_neg','sent_neu']].fillna(1/3)

# 建立ChromaDB向量库
client    = chromadb.Client()
market_db = client.create_collection(
    name="market_k_test",
    metadata={"hnsw:space": "cosine"}
)
market_db.add(
    embeddings=train_features.tolist(),
    documents=[f"day_{i}" for i in range(len(train_stock))],
    ids=[f"day_{i}" for i in range(len(train_stock))]
)
print(f"向量库建立完成！共{market_db.count()}条")

# ===== 定义模型和训练函数 =====
class RAGDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim=3, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.state_embed  = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(act_dim, d_model)
        self.rtg_embed    = nn.Linear(1, d_model)
        self.pos_embed    = nn.Embedding(500, d_model)
        self.ln           = nn.LayerNorm(d_model)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head  = nn.Linear(d_model, act_dim)

    def forward(self, states, actions, rtgs):
        B, T, _ = states.shape
        s = self.state_embed(states)
        r = self.rtg_embed(rtgs.unsqueeze(-1))
        a_shift = torch.zeros_like(actions)
        a_shift[:, 1:] = actions[:, :-1]
        a   = self.action_embed(a_shift)
        pos = torch.arange(T).unsqueeze(0).expand(B, -1).to(states.device)
        p   = self.pos_embed(pos)
        x   = self.ln(s + a + r + p)
        x   = self.transformer(x)
        return self.action_head(x[:, -1, :])

def compute_rtg(rewards):
    rtg = np.zeros(len(rewards))
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + cumulative
        rtg[i] = cumulative
    return rtg

def backtest(prices, actions, initial_cash=10000):
    cash, shares, portfolio = initial_cash, 0, []
    for i in range(len(prices)):
        price = prices[i]
        if actions[i] == 2 and cash > 0:
            shares, cash = cash / price, 0
        elif actions[i] == 0 and shares > 0:
            cash, shares = shares * price, 0
        portfolio.append(cash + shares * price)
    return np.array(portfolio)

def calc_sharpe(portfolio):
    daily_ret = np.diff(portfolio) / portfolio[:-1]
    return (daily_ret.mean() - 0.03/252) / (daily_ret.std() + 1e-8) * np.sqrt(252)

def calc_maxdd(portfolio):
    peak = np.maximum.accumulate(portfolio)
    return ((portfolio - peak) / peak).min() * 100

# ===== 对不同K值进行实验 =====
K_values = [1, 3, 5, 10]
results  = {}

for K in K_values:
    print(f"\n=== 测试 K={K} ===")

    # 生成RAG向量
    rag_vectors = []
    for i in range(len(test_stock)):
        query_vec    = test_features[i:i+1]
        results_db   = market_db.query(
            query_embeddings=query_vec.tolist(),
            n_results=K
        )
        retrieved_ids  = [int(doc.split('_')[1]) for doc in results_db['ids'][0]]
        retrieved_vecs = train_features[retrieved_ids]
        rag_vec        = retrieved_vecs.mean(axis=0)
        rag_vectors.append(rag_vec)

    rag_vectors = np.array(rag_vectors)

    # 融合特征
    sent_feats   = test_stock[['sent_pos','sent_neg','sent_neu']].values
    all_features = np.concatenate([test_features, sent_feats, rag_vectors], axis=1)
    state_dim    = all_features.shape[1]

    # 动作标签
    actions_arr = []
    for i in range(len(test_stock)):
        rsi = test_stock['RSI'].iloc[i]
        if rsi < 50:
            actions_arr.append(2)
        elif rsi > 50:
            actions_arr.append(0)
        else:
            actions_arr.append(1)
    test_stock['Action'] = actions_arr
    test_stock['Reward'] = test_stock['Close'].pct_change().fillna(0)
    test_stock['RTG']    = compute_rtg(test_stock['Reward'].values)
    rtg_min = test_stock['RTG'].min()
    rtg_max = test_stock['RTG'].max()
    test_stock['RTG_scaled'] = (test_stock['RTG'] - rtg_min) / (rtg_max - rtg_min + 1e-8)

    SEQ_LEN  = 20
    act_arr  = test_stock['Action'].values
    rtg_arr  = test_stock['RTG_scaled'].values

    X_states, X_actions, X_rtgs, y_actions = [], [], [], []
    for i in range(len(test_stock) - SEQ_LEN):
        X_states.append(all_features[i:i+SEQ_LEN])
        X_actions.append(act_arr[i:i+SEQ_LEN])
        X_rtgs.append(rtg_arr[i:i+SEQ_LEN])
        y_actions.append(act_arr[i+SEQ_LEN])

    X_states  = np.array(X_states,  dtype=np.float32)
    X_actions = np.array(X_actions, dtype=np.int64)
    X_rtgs    = np.array(X_rtgs,    dtype=np.float32)
    y_actions = np.array(y_actions, dtype=np.int64)

    split = int(len(X_states) * 0.8)

    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_states[:split]).to(device),
        torch.LongTensor(X_actions[:split]).to(device),
        torch.FloatTensor(X_rtgs[:split]).to(device),
        torch.LongTensor(y_actions[:split]).to(device)
    ), batch_size=64, shuffle=True)

    test_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_states[split:]).to(device),
        torch.LongTensor(X_actions[split:]).to(device),
        torch.FloatTensor(X_rtgs[split:]).to(device),
        torch.LongTensor(y_actions[split:]).to(device)
    ), batch_size=64, shuffle=False)

    # 训练
    model     = RAGDecisionTransformer(state_dim=state_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss = float('inf')

    for epoch in range(15):
        model.train()
        for s, a, r, y in train_loader:
            optimizer.zero_grad()
            pred = model(s, a, r)
            loss = criterion(pred, y)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for s, a, r, y in test_loader:
                pred      = model(s, a, r)
                loss      = criterion(pred, y)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f'best_k{K}.pth')

    # 回测
    model.load_state_dict(torch.load(f'best_k{K}.pth', weights_only=True))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for s, a, r, y in test_loader:
            pred = model(s, a, r)
            all_preds.extend(pred.argmax(dim=-1).cpu().numpy())

    all_preds   = np.array(all_preds)
    test_prices = test_stock['Close'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]
    portfolio   = backtest(test_prices, all_preds)
    sharpe      = calc_sharpe(portfolio)
    maxdd       = calc_maxdd(portfolio)

    results[K] = {'sharpe': sharpe, 'maxdd': maxdd}
    print(f"K={K} | 夏普比率：{sharpe:.4f} | 最大回撤：{maxdd:.2f}%")

# ===== 可视化 =====
print("\n=== K值敏感性分析结果 ===")
for K, v in results.items():
    print(f"K={K:2d} | 夏普比率：{v['sharpe']:.4f} | 最大回撤：{v['maxdd']:.2f}%")

K_list     = list(results.keys())
sharpe_list = [results[K]['sharpe'] for K in K_list]
maxdd_list  = [results[K]['maxdd'] for K in K_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(K_list, sharpe_list, marker='o', linewidth=2, color='steelblue')
ax1.set_title('Sharpe Ratio vs K')
ax1.set_xlabel('K (Number of Retrieved Scenarios)')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_xticks(K_list)
ax1.grid(True, alpha=0.3)

ax2.plot(K_list, maxdd_list, marker='o', linewidth=2, color='orange')
ax2.set_title('Maximum Drawdown vs K')
ax2.set_xlabel('K (Number of Retrieved Scenarios)')
ax2.set_ylabel('Maximum Drawdown (%)')
ax2.set_xticks(K_list)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('k_sensitivity.png', dpi=100, bbox_inches='tight')
plt.show()
print("图表已保存！")