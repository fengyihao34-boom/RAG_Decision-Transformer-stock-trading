import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# ===== 第一部分：测试ChromaDB + Sentence-BERT =====
print("=== 测试RAG基础组件 ===")

# 加载Sentence-BERT模型
print("加载Sentence-BERT...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT加载成功！")

# 测试编码几条金融新闻
test_news = [
    "Stock market crashes amid recession fears",
    "Tech stocks rally on strong earnings report",
    "Federal Reserve raises interest rates again",
    "Amazon reports record quarterly revenue",
    "Oil prices surge due to supply concerns"
]

# 编码成向量
embeddings = sbert.encode(test_news)
print(f"\n编码结果：")
print(f"新闻数量：{len(test_news)}条")
print(f"每条向量维度：{embeddings.shape[1]}")

# 测试ChromaDB
print("\n=== 测试ChromaDB ===")
client = chromadb.Client()
collection = client.create_collection("test_market")

# 存入向量
collection.add(
    embeddings=embeddings.tolist(),
    documents=test_news,
    ids=[f"news_{i}" for i in range(len(test_news))]
)
print(f"成功存入 {collection.count()} 条记录")

# 检索最相似的2条
query = "Market downturn and economic crisis"
query_embedding = sbert.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

print(f"\n查询：'{query}'")
print(f"最相似的2条新闻：")
for i, doc in enumerate(results['documents'][0]):
    dist = results['distances'][0][i]
    print(f"  {i+1}. {doc}")
    print(f"     相似度距离：{dist:.4f}")

print("\nRAG基础组件测试成功！")

# ===== 第二部分：构建历史情景向量库 =====

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("\n=== 构建历史情景向量库 ===")

# 加载股票数据
df = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
stock = df[df['Symbol'] == 'GOOG'].copy().reset_index(drop=True)

# 计算技术指标
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

# 加载真实新闻数据
df_news = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_real_news.csv')
df_news['date'] = pd.to_datetime(df_news['date'])

# 合并新闻到股票数据
# 按日期聚合：每天取第一条新闻标题
daily_news = df_news.groupby('date')['title'].first().reset_index()
daily_news.columns = ['Date', 'news_title']

stock = stock.merge(daily_news, on='Date', how='left')

# 没有新闻的日期用通用文本填充
stock['news_title'] = stock['news_title'].fillna('Stock market trading day, no major news')

print(f"股票数据：{len(stock)}天")
print(f"有真实新闻的天数：{stock['news_title'].notna().sum()}天")
print(f"时间范围：{stock['Date'].min().date()} 到 {stock['Date'].max().date()}")

# 划分训练集（用来建向量库）和测试集
split_date = '2022-01-01'
train_stock = stock[stock['Date'] < split_date].reset_index(drop=True)
test_stock  = stock[stock['Date'] >= split_date].reset_index(drop=True)

print(f"\n训练集（建库用）：{len(train_stock)}天")
print(f"测试集（检索用）：{len(test_stock)}天")

# 用技术指标特征作为历史情景向量
print("\n构建技术指标特征向量...")
feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']

scaler = MinMaxScaler()
# 只用训练集拟合scaler
train_features = scaler.fit_transform(
    train_stock[feature_cols].fillna(0).values
)
print(f"特征向量形状：{train_features.shape}")
# 输出应该是 (3002, 9)

# 建立ChromaDB向量库
print("\n建立ChromaDB向量库...")
client2    = chromadb.Client()
market_db  = client2.create_collection(
    name="market_history",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 存入所有训练集向量
market_db.add(
    embeddings=train_features.tolist(),
    documents=[f"day_{i}" for i in range(len(train_stock))],
    ids=[f"day_{i}" for i in range(len(train_stock))],
    metadatas=[{
        'date':   str(train_stock['Date'].iloc[i].date()),
        'close':  float(train_stock['Close'].iloc[i]),
        'rsi':    float(train_stock['RSI'].iloc[i])
    } for i in range(len(train_stock))]
)

print(f"向量库建立完成！共存入 {market_db.count()} 条历史情景")

# 测试检索
test_query    = "Amazon stock drops amid market selloff"
# 用测试集第一天的技术指标作为查询向量
test_features = scaler.transform(
    test_stock[feature_cols].fillna(0).values
)
query_vec = test_features[0:1]
results       = market_db.query(
    query_embeddings=query_vec.tolist(),
    n_results=3
)

print(f"\n测试检索：'{test_query}'")
print(f"最相似的3个历史情景：")
for i in range(3):
    print(f"  {i+1}. 日期：{results['metadatas'][0][i]['date']}")
    print(f"     新闻：{results['documents'][0][i][:60]}...")
    print(f"     距离：{results['distances'][0][i]:.4f}")

# ===== 第三部分：为每天生成RAG检索向量 =====

print("\n=== 生成RAG检索向量 ===")

K = 3  # 检索最相似的3个历史情景

# 对测试集每一天进行检索
rag_vectors = []

for i in range(len(test_stock)):
    # 取当天的技术指标向量作为查询
    query_vec = test_features[i:i+1]

    # 检索K个最相似的历史情景
    results = market_db.query(
        query_embeddings=query_vec.tolist(),
        n_results=K
    )

    # 取K个历史向量的ID，再从train_features里取出对应向量
    retrieved_ids = [int(doc.split('_')[1]) for doc in results['ids'][0]]
    retrieved_vecs = train_features[retrieved_ids]

    # 取平均得到RAG检索向量
    rag_vec = retrieved_vecs.mean(axis=0)
    rag_vectors.append(rag_vec)

    if (i+1) % 100 == 0:
        print(f"已处理 {i+1}/{len(test_stock)} 天...")

rag_vectors = np.array(rag_vectors)
print(f"\nRAG检索向量生成完成！")
print(f"形状：{rag_vectors.shape}")  # 应该是(747, 9)
print(f"第一天RAG向量（前5维）：{rag_vectors[0][:5].round(4)}")

# 保存RAG向量
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/rag_vectors.npy', rag_vectors)
print("RAG向量已保存！")

# ===== 第四部分：RAG-DT完整训练和回测 =====

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

print("\n=== RAG-DT完整训练 ===")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载之前保存的RAG向量
rag_vectors = np.load(
    '/Users/fengyihao/PyCharmMiscProject/深度学习/data/rag_vectors.npy'
)

# 加载真实FinBERT情感数据（day10_real_news.py里保存的）
df_sentiment = pd.read_csv(
    '/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_sentiment.csv'
)
df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])

# 合并情感数据到测试集
test_stock = test_stock.reset_index(drop=True)
test_stock = test_stock.merge(df_sentiment, on='Date', how='left')

# 没有情感数据的日期用均匀分布填充
test_stock[['sent_pos','sent_neg','sent_neu']] = \
    test_stock[['sent_pos','sent_neg','sent_neu']].ffill().fillna(1/3)

print(f"测试集情感数据覆盖率：{test_stock['sent_pos'].notna().mean()*100:.1f}%")

# 归一化价格特征
scaler_test  = MinMaxScaler()
price_feats  = scaler_test.fit_transform(test_stock[feature_cols].values)
sent_feats   = test_stock[['sent_pos','sent_neg','sent_neu']].values
all_features = np.concatenate([price_feats, sent_feats, rag_vectors], axis=1)

print(f"增强状态维度：{all_features.shape[1]}维")

# 动作标签
actions = []
for i in range(len(test_stock)):
    rsi = test_stock['RSI'].iloc[i]
    if rsi < 50:
        actions.append(2)
    elif rsi > 50:
        actions.append(0)
    else:
        actions.append(1)
test_stock['Action'] = actions
test_stock['Reward'] = test_stock['Close'].pct_change().fillna(0)

# 计算RTG
def compute_rtg(rewards):
    rtg = np.zeros(len(rewards))
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + cumulative
        rtg[i] = cumulative
    return rtg

test_stock['RTG'] = compute_rtg(test_stock['Reward'].values)
rtg_min = test_stock['RTG'].min()
rtg_max = test_stock['RTG'].max()
test_stock['RTG_scaled'] = (test_stock['RTG'] - rtg_min) / (rtg_max - rtg_min + 1e-8)

# 构建序列数据集
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

print(f"训练集：{split}个样本，测试集：{len(X_states)-split}个样本")

# 定义RAG-DT模型（state_dim=21）
class RAGDecisionTransformer(nn.Module):
    def __init__(self, state_dim=21, act_dim=3, d_model=64, nhead=4, num_layers=2):
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

model     = RAGDecisionTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
# 改成
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.3
)

print(f"RAG-DT模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# 训练
EPOCHS    = 15
best_loss = float('inf')
train_losses, test_losses, test_accs = [], [], []

print("\n开始训练RAG-DT...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for s, a, r, y in train_loader:
        optimizer.zero_grad()
        pred = model(s, a, r)
        loss = criterion(pred, y)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for s, a, r, y in test_loader:
            pred      = model(s, a, r)
            loss      = criterion(pred, y)
            test_loss += loss.item()
            predicted  = pred.argmax(dim=-1)
            correct   += (predicted == y).sum().item()
            total     += y.size(0)
    test_loss /= len(test_loader)
    test_acc   = correct / total * 100

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_rag_dt.pth')

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | "
              f"Test: {test_loss:.4f} | "
              f"Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试损失：{best_loss:.4f}")

# 回测
print("\n=== 回测对比 ===")
model.load_state_dict(torch.load('best_rag_dt.pth', weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for s, a, r, y in test_loader:
        pred = model(s, a, r)
        all_preds.extend(pred.argmax(dim=-1).cpu().numpy())

all_preds   = np.array(all_preds)
test_prices = test_stock['Close'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]
test_dates  = test_stock['Date'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]

print(f"预测动作分布：")
print(f"  卖出(0)：{(all_preds==0).sum()}次")
print(f"  持有(1)：{(all_preds==1).sum()}次")
print(f"  买入(2)：{(all_preds==2).sum()}次")

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

def calc_metrics(portfolio):
    daily_ret  = np.diff(portfolio) / portfolio[:-1]
    total_ret  = (portfolio[-1] - portfolio[0]) / portfolio[0]
    n_years    = len(portfolio) / 252
    annual_ret = (1 + total_ret) ** (1/n_years) - 1
    sharpe     = (daily_ret.mean() - 0.03/252) / (daily_ret.std() + 1e-8) * np.sqrt(252)
    peak       = np.maximum.accumulate(portfolio)
    max_dd     = ((portfolio - peak) / peak).min()
    return total_ret*100, annual_ret*100, sharpe, max_dd*100

rag_portfolio = backtest(test_prices, all_preds)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/rag_port.npy', rag_portfolio)
bh_portfolio  = backtest(test_prices, np.array([2]+[1]*(len(test_prices)-1)))

rag_m = calc_metrics(rag_portfolio)
bh_m  = calc_metrics(bh_portfolio)

print(f"\n{'指标':<15} {'RAG-DT':>12} {'Buy&Hold':>12}")
print("-" * 42)
print(f"{'总收益率':<15} {rag_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {rag_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {rag_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {rag_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

# 可视化
plt.figure(figsize=(12, 5))
dates = pd.to_datetime(test_dates[:len(rag_portfolio)])
plt.plot(dates, rag_portfolio, label='RAG-DT',    linewidth=1.5, color='steelblue')
plt.plot(dates, bh_portfolio,  label='Buy & Hold', linewidth=1.5, color='orange')
plt.title('Portfolio Value: RAG-DT vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rag_dt_backtest.png', dpi=100, bbox_inches='tight')
plt.show()
print("回测图已保存！")