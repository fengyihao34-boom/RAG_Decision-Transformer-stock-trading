import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# ===== 加载数据 =====
df    = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
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

# 归一化
feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']
scaler       = MinMaxScaler()
stock_scaled = stock.copy()
stock_scaled[feature_cols] = scaler.fit_transform(stock[feature_cols])

# ===== 生成动作标签（只用RSI，确保均衡）=====
actions = []
for i in range(len(stock)):
    rsi = stock['RSI'].iloc[i]
    if rsi < 50:
        actions.append(2)   # 买入
    elif rsi > 50:
        actions.append(0)   # 卖出
    else:
        actions.append(1)   # 持有

stock['Action'] = actions
stock['Reward'] = stock['Close'].pct_change().fillna(0)

print(f"动作分布：")
print(f"  卖出(0)：{(stock['Action']==0).sum()}次 ({(stock['Action']==0).mean()*100:.1f}%)")
print(f"  持有(1)：{(stock['Action']==1).sum()}次 ({(stock['Action']==1).mean()*100:.1f}%)")
print(f"  买入(2)：{(stock['Action']==2).sum()}次 ({(stock['Action']==2).mean()*100:.1f}%)")

# ===== 计算RTG =====
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

# ===== 构建序列数据集 =====
SEQ_LEN  = 20
features = stock_scaled[feature_cols].values
actions  = stock['Action'].values
rtgs     = stock['RTG_scaled'].values

X_states, X_actions, X_rtgs, y_actions = [], [], [], []
for i in range(len(stock) - SEQ_LEN):
    X_states.append(features[i:i+SEQ_LEN])
    X_actions.append(actions[i:i+SEQ_LEN])
    X_rtgs.append(rtgs[i:i+SEQ_LEN])
    y_actions.append(actions[i+SEQ_LEN])

X_states  = np.array(X_states,  dtype=np.float32)
X_actions = np.array(X_actions, dtype=np.int64)
X_rtgs    = np.array(X_rtgs,    dtype=np.float32)
y_actions = np.array(y_actions, dtype=np.int64)

split  = int(len(X_states) * 0.8)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

print(f"训练集：{split}个样本，测试集：{len(X_states)-split}个样本")

# ===== 定义DT模型 =====
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=9, act_dim=3, d_model=64, nhead=4, num_layers=2):
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

model     = DecisionTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"使用设备：{device}")

# ===== 训练 =====
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
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_dt_simple.pth')

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试损失：{best_loss:.4f}")

# 可视化训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train')
ax1.plot(test_losses,  label='Test')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(test_accs, color='green')
ax2.set_title('Test Accuracy')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dt_simple_training.png', dpi=100)
plt.show()

# ===== 回测评估 =====
print("\n=== 回测评估 ===")

model.load_state_dict(torch.load('best_dt_simple.pth', weights_only=True))
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

dt_portfolio = backtest(test_prices, all_preds)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/dt_port.npy', dt_portfolio)
bh_actions   = np.array([2] + [1]*(len(test_prices)-1))
bh_portfolio = backtest(test_prices, bh_actions)

dt_m = calc_metrics(dt_portfolio)
bh_m = calc_metrics(bh_portfolio)

print(f"\n{'指标':<15} {'DT策略':>12} {'Buy&Hold':>12}")
print("-" * 40)
print(f"{'总收益率':<15} {dt_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {dt_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {dt_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {dt_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

plt.figure(figsize=(12, 5))
dates = pd.to_datetime(test_dates[:len(dt_portfolio)])
plt.plot(dates, dt_portfolio, label='DT Strategy', linewidth=1.5)
plt.plot(dates, bh_portfolio, label='Buy & Hold',  linewidth=1.5)
plt.title('Portfolio Value: DT vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dt_backtest.png', dpi=100, bbox_inches='tight')
plt.show()
print("回测图已保存！")