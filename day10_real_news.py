import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ===== 第一部分：加载并处理真实新闻数据 =====
print("\n=== 处理真实新闻数据 ===")

# 加载新闻数据
df_news = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_real_news.csv')
df_news['date'] = pd.to_datetime(df_news['date'])

# 转换情感标签
def convert_sentiment(label):
    label = str(label).lower()
    if 'bullish' in label:
        return 'positive'
    elif 'bearish' in label:
        return 'negative'
    else:
        return 'neutral'

df_news['sentiment_clean'] = df_news['sentiment'].apply(convert_sentiment)

# 按日期聚合（每天可能有多条新闻，取平均分数）
df_news['score_signed'] = df_news['score'].copy()
df_news.loc[df_news['sentiment_clean']=='negative', 'score_signed'] *= -1

daily_sentiment = df_news.groupby('date').agg(
    avg_score=('score_signed', 'mean'),
    news_count=('title', 'count'),
    dominant_sentiment=('sentiment_clean', lambda x: x.mode()[0])
).reset_index()

print(f"有新闻数据的交易日：{len(daily_sentiment)}天")
print(f"时间范围：{daily_sentiment['date'].min()} 到 {daily_sentiment['date'].max()}")
print(f"\n情感分布：")
print(daily_sentiment['dominant_sentiment'].value_counts())

# ===== 第二部分：用FinBERT提取真实情感向量 =====
print("\n=== 用FinBERT提取真实情感向量 ===")

model_name = "ProsusAI/finbert"
tokenizer  = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
finbert    = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
finbert    = finbert.to(device)
finbert.eval()
print("FinBERT加载成功！")

# 对每条新闻提取情感向量，然后按日期平均
print("提取新闻情感向量...")
sent_vectors = []
for idx, row in df_news.iterrows():
    inputs = tokenizer(
        str(row['title'])[:256],
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = finbert(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    sent_vectors.append({
        'date':     row['date'],
        'sent_pos': probs[0],
        'sent_neg': probs[1],
        'sent_neu': probs[2]
    })
    if (idx+1) % 50 == 0:
        print(f"已处理 {idx+1}/300 条新闻...")

df_sent = pd.DataFrame(sent_vectors)
df_sent_daily = df_sent.groupby('date').agg({
    'sent_pos': 'mean',
    'sent_neg': 'mean',
    'sent_neu': 'mean'
}).reset_index()

print(f"\n每日情感向量统计：")
print(f"平均正面：{df_sent_daily['sent_pos'].mean():.3f}")
print(f"平均负面：{df_sent_daily['sent_neg'].mean():.3f}")
print(f"平均中性：{df_sent_daily['sent_neu'].mean():.3f}")

# ===== 第三部分：和股价数据对齐 =====
print("\n=== 对齐股价和情感数据 ===")

df_stock = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
stock = df_stock[df_stock['Symbol'] == 'GOOG'].copy().reset_index(drop=True)

# 只取2022-2024的数据（和新闻数据对齐）
stock = stock[stock['Date'] >= '2022-01-01'].reset_index(drop=True)

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

# 合并情感数据
stock = stock.merge(df_sent_daily, left_on='Date', right_on='date', how='left')

# 对没有新闻的日子用前向填充
stock[['sent_pos','sent_neg','sent_neu']] = stock[['sent_pos','sent_neg','sent_neu']].ffill()
stock[['sent_pos','sent_neg','sent_neu']] = stock[['sent_pos','sent_neg','sent_neu']].fillna(1/3)  # 剩余用均匀分布
stock[['Date','sent_pos','sent_neg','sent_neu']].to_csv(
    '/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_sentiment.csv',
    index=False
)
print("情感数据已保存到 goog_sentiment.csv！")
print(f"合并后数据量：{len(stock)}天")
print(f"有真实新闻的天数：{stock['sent_pos'].notna().sum()}天")
print(f"时间范围：{stock['Date'].min().date()} 到 {stock['Date'].max().date()}")

# ===== 第四部分：训练DT+真实情感 =====
print("\n=== 训练DT + 真实情感 ===")

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
print(f"训练集：{split}个，测试集：{len(X_states)-split}个")

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

model     = DecisionTransformerWithSentiment().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

EPOCHS    = 30
best_loss = float('inf')
train_losses, test_losses, test_accs = [], [], []

print("\n开始训练...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for s, a, r, y in train_loader:
        optimizer.zero_grad()
        pred = model(s, a, r)
        loss = criterion(pred, y)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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
    scheduler.step(test_loss)
    test_acc   = correct / total * 100
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_dt_real_sentiment.pth')

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳损失：{best_loss:.4f}")

# ===== 回测 =====
print("\n=== 回测对比 ===")
model.load_state_dict(torch.load('best_dt_real_sentiment.pth', weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for s, a, r, y in test_loader:
        pred = model(s, a, r)
        all_preds.extend(pred.argmax(dim=-1).cpu().numpy())

all_preds   = np.array(all_preds)
test_prices = stock['Close'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]
test_dates  = stock['Date'].values[split+SEQ_LEN:split+SEQ_LEN+len(all_preds)]

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

sent_portfolio = backtest(test_prices, all_preds)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/dt_sent_port.npy', sent_portfolio)
bh_portfolio   = backtest(test_prices, np.array([2]+[1]*(len(test_prices)-1)))

sent_m = calc_metrics(sent_portfolio)
bh_m   = calc_metrics(bh_portfolio)

print(f"\n{'指标':<15} {'DT+真实情感':>12} {'Buy&Hold':>12}")
print("-" * 42)
print(f"{'总收益率':<15} {sent_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {sent_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {sent_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {sent_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses, label='Test')
axes[0].set_title('Loss Curve (DT + Real Sentiment)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

dates = pd.to_datetime(test_dates[:len(sent_portfolio)])
axes[1].plot(dates, sent_portfolio, label='DT+Real Sentiment', linewidth=1.5)
axes[1].plot(dates, bh_portfolio,  label='Buy & Hold',        linewidth=1.5)
axes[1].set_title('Portfolio: DT+Real Sentiment vs Buy&Hold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_real_news.png', dpi=100)
plt.show()
print("图表已保存！")