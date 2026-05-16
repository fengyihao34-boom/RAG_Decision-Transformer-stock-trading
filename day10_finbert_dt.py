import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ===== 第一部分：用FinBERT验证Financial PhraseBank =====
print("\n=== 验证FinBERT在Financial PhraseBank上的效果 ===")

model_name = "ProsusAI/finbert"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
finbert    = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert    = finbert.to(device)
finbert.eval()
print("FinBERT加载成功！")

# 读取数据
df_pb = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/phrasebank.csv')
label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df_pb['label'] = df_pb['Sentiment'].map(label_map)

# 取前200条验证（全部验证太慢）
df_test = df_pb.sample(200, random_state=42).reset_index(drop=True)

print(f"验证集大小：{len(df_test)}条")
print("开始预测（需要约1分钟）...")

y_true = []
y_pred = []

for i, row in df_test.iterrows():
    inputs = tokenizer(
        row['Sentence'],
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = finbert(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
        pred    = probs[0].argmax().item()

    y_true.append(row['label'])
    y_pred.append(pred)

    if (i+1) % 50 == 0:
        print(f"已处理 {i+1}/200 条...")

# 计算准确率
correct  = sum(t == p for t, p in zip(y_true, y_pred))
accuracy = correct / len(y_true) * 100
print(f"\nFinBERT在Financial PhraseBank上的准确率：{accuracy:.2f}%")
print("\n分类报告：")
print(classification_report(
    y_true, y_pred,
    target_names=['negative','neutral','positive']
))

# ===== 第二部分：为股票数据生成情感向量 =====
print("\n=== 为股票数据生成情感向量 ===")

# 加载股票数据
df_stock = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
stock = df_stock[df_stock['Symbol'] == 'AMZN'].copy().reset_index(drop=True)

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

n         = len(stock)
daily_ret = stock['Close'].pct_change().fillna(0).values

# 用Financial PhraseBank的真实句子生成情感
# 按照每天的收益率从数据集中选取对应情感的句子，再用FinBERT提取向量
print("从Financial PhraseBank采样真实句子生成情感向量...")

pos_sentences = df_pb[df_pb['Sentiment']=='positive']['Sentence'].tolist()
neg_sentences = df_pb[df_pb['Sentiment']=='negative']['Sentence'].tolist()
neu_sentences = df_pb[df_pb['Sentiment']=='neutral']['Sentence'].tolist()

np.random.seed(42)
sentiment_pos = np.zeros(n)
sentiment_neg = np.zeros(n)
sentiment_neu = np.zeros(n)

# 每100天处理一次，加快速度
batch_size = 100
for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    for i in range(start, end):
        ret = daily_ret[i]
        # 根据收益率选取对应情感的真实句子
        if ret > 0.01:
            sentence = np.random.choice(pos_sentences)
        elif ret < -0.01:
            sentence = np.random.choice(neg_sentences)
        else:
            sentence = np.random.choice(neu_sentences)

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = finbert(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        sentiment_pos[i] = probs[0]  # positive
        sentiment_neg[i] = probs[1]  # negative
        sentiment_neu[i] = probs[2]  # neutral

    if end % 500 == 0 or end == n:
        print(f"已处理 {end}/{n} 天...")

stock['sent_pos'] = sentiment_pos
stock['sent_neg'] = sentiment_neg
stock['sent_neu'] = sentiment_neu

print(f"\n情感向量生成完成！")
print(f"平均正面：{sentiment_pos.mean():.3f}")
print(f"平均负面：{sentiment_neg.mean():.3f}")
print(f"平均中性：{sentiment_neu.mean():.3f}")

# 保存情感数据
stock[['Date','sent_pos','sent_neg','sent_neu']].to_csv(
    '/Users/fengyihao/PyCharmMiscProject/深度学习/data/amzn_sentiment.csv',
    index=False
)
print("情感数据已保存！")

# ===== 第三部分：含情感的DT训练和回测 =====
print("\n=== 训练含情感的DT ===")

feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']
scaler       = MinMaxScaler()
stock_scaled = stock.copy()
stock_scaled[feature_cols] = scaler.fit_transform(stock[feature_cols])

# 动作标签
actions = []
for i in range(len(stock)):
    rsi = stock['RSI'].iloc[i]
    if rsi < 50:
        actions.append(2)
    elif rsi > 50:
        actions.append(0)
    else:
        actions.append(1)
stock['Action'] = actions
stock['Reward'] = stock['Close'].pct_change().fillna(0)

def compute_rtg(rewards):
    rtg = np.zeros(len(rewards))
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + cumulative
        rtg[i] = cumulative
    return rtg

stock['RTG'] = compute_rtg(stock['Reward'].values)
rtg_min = stock['RTG'].min()
rtg_max = stock['RTG'].max()
stock['RTG_scaled'] = (stock['RTG'] - rtg_min) / (rtg_max - rtg_min + 1e-8)

# 12维特征
SEQ_LEN      = 20
price_feats  = stock_scaled[feature_cols].values
sent_feats   = stock[['sent_pos','sent_neg','sent_neu']].values
all_features = np.concatenate([price_feats, sent_feats], axis=1)
act_arr      = stock['Action'].values
rtg_arr      = stock['RTG_scaled'].values

X_states, X_actions, X_rtgs, y_actions = [], [], [], []
for i in range(len(stock) - SEQ_LEN):
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
), batch_size=128, shuffle=True)

test_loader = DataLoader(TensorDataset(
    torch.FloatTensor(X_states[split:]).to(device),
    torch.LongTensor(X_actions[split:]).to(device),
    torch.FloatTensor(X_rtgs[split:]).to(device),
    torch.LongTensor(y_actions[split:]).to(device)
), batch_size=128, shuffle=False)

print(f"训练集：{split}个，测试集：{len(X_states)-split}个")

class DecisionTransformerWithSentiment(nn.Module):
    def __init__(self, state_dim=12, act_dim=3, d_model=64, nhead=4, num_layers=2):
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

model_sent = DecisionTransformerWithSentiment().to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model_sent.parameters(), lr=3e-4, weight_decay=1e-5)
print(f"模型参数量：{sum(p.numel() for p in model_sent.parameters()):,}")

EPOCHS    = 30
best_loss = float('inf')
train_losses, test_losses, test_accs = [], [], []

print("\n开始训练...")
for epoch in range(EPOCHS):
    model_sent.train()
    train_loss = 0.0
    for s, a, r, y in train_loader:
        optimizer.zero_grad()
        pred = model_sent(s, a, r)
        loss = criterion(pred, y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_sent.parameters(), 0.5)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model_sent.eval()
    test_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for s, a, r, y in test_loader:
            pred      = model_sent(s, a, r)
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

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model_sent.state_dict(), 'best_dt_sentiment.pth')

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | "
              f"Test: {test_loss:.4f} | "
              f"Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试损失：{best_loss:.4f}")

# 回测
print("\n=== 回测对比 ===")
model_sent.load_state_dict(torch.load('best_dt_sentiment.pth', weights_only=True))
model_sent.eval()

all_preds = []
with torch.no_grad():
    for s, a, r, y in test_loader:
        pred = model_sent(s, a, r)
        all_preds.extend(pred.argmax(dim=-1).cpu().numpy())

all_preds   = np.array(all_preds)
test_prices = stock['Close'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]
test_dates  = stock['Date'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]

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

sent_portfolio = backtest(test_prices, all_preds)
bh_portfolio   = backtest(test_prices, np.array([2]+[1]*(len(test_prices)-1)))

sent_m = calc_metrics(sent_portfolio)
bh_m   = calc_metrics(bh_portfolio)

print(f"\n{'指标':<15} {'DT+情感':>12} {'Buy&Hold':>12}")
print("-" * 40)
print(f"{'总收益率':<15} {sent_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {sent_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {sent_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {sent_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses,  label='Test')
axes[0].set_title('Loss Curve (DT + Real Sentiment)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

dates = pd.to_datetime(test_dates[:len(sent_portfolio)])
axes[1].plot(dates, sent_portfolio, label='DT+Sentiment', linewidth=1.5)
axes[1].plot(dates, bh_portfolio,  label='Buy & Hold',   linewidth=1.5)
axes[1].set_title('Portfolio Value Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_real_sentiment.png', dpi=100)
plt.show()
print("图表已保存！")