import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

print("=== LSTM Baseline ===")

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

# 按日期划分训练集和测试集
train_stock = stock[stock['Date'] < '2022-01-01'].reset_index(drop=True)
test_stock  = stock[stock['Date'] >= '2022-01-01'].reset_index(drop=True)

print(f"训练集：{len(train_stock)}天，测试集：{len(test_stock)}天")

# 归一化（只用训练集拟合）
feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']
scaler       = MinMaxScaler()
train_scaled = train_stock.copy()
test_scaled  = test_stock.copy()
train_scaled[feature_cols] = scaler.fit_transform(train_stock[feature_cols])
test_scaled[feature_cols]  = scaler.transform(test_stock[feature_cols])

# 动作标签
def get_actions(df):
    actions = []
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        if rsi < 50:
            actions.append(2)
        elif rsi > 50:
            actions.append(0)
        else:
            actions.append(1)
    return np.array(actions)

train_acts = get_actions(train_stock)
test_acts  = get_actions(test_stock)

# 构建序列数据集
SEQ_LEN        = 20
train_features = train_scaled[feature_cols].values
test_features  = test_scaled[feature_cols].values

X_train, y_train = [], []
for i in range(len(train_stock) - SEQ_LEN):
    X_train.append(train_features[i:i+SEQ_LEN])
    y_train.append(train_acts[i+SEQ_LEN])

X_test, y_test = [], []
for i in range(len(test_stock) - SEQ_LEN):
    X_test.append(test_features[i:i+SEQ_LEN])
    y_test.append(test_acts[i+SEQ_LEN])

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
X_test  = np.array(X_test,  dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.int64)

train_loader = DataLoader(TensorDataset(
    torch.FloatTensor(X_train).to(device),
    torch.LongTensor(y_train).to(device)
), batch_size=128, shuffle=True)

test_loader = DataLoader(TensorDataset(
    torch.FloatTensor(X_test).to(device),
    torch.LongTensor(y_test).to(device)
), batch_size=128, shuffle=False)

print(f"训练序列：{len(X_train)}个，测试序列：{len(X_test)}个")

# ===== 定义LSTM模型 =====
class LSTMTrader(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model     = LSTMTrader().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"LSTM模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# ===== 训练 =====
EPOCHS    = 30
best_loss = float('inf')
train_losses, test_losses = [], []

print("\n开始训练LSTM...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred      = model(x)
            loss      = criterion(pred, y)
            test_loss += loss.item()
            predicted  = pred.argmax(dim=-1)
            correct   += (predicted == y).sum().item()
            total     += y.size(0)
    test_loss /= len(test_loader)
    test_acc   = correct / total * 100

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_lstm.pth')

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | "
              f"Test: {test_loss:.4f} | "
              f"Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试损失：{best_loss:.4f}")

# ===== 回测 =====
print("\n=== 回测对比 ===")
model.load_state_dict(torch.load('best_lstm.pth', weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x)
        all_preds.extend(pred.argmax(dim=-1).cpu().numpy())

all_preds   = np.array(all_preds)
test_prices = test_stock['Close'].values[SEQ_LEN:SEQ_LEN+len(all_preds)]
test_dates  = test_stock['Date'].values[SEQ_LEN:SEQ_LEN+len(all_preds)]

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

lstm_port = backtest(test_prices, all_preds)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/lstm_port.npy', lstm_port)
bh_port   = backtest(
    test_prices, np.array([2]+[1]*(len(test_prices)-1))
)

lstm_m = calc_metrics(lstm_port)
bh_m   = calc_metrics(bh_port)

print(f"\n{'指标':<15} {'LSTM':>12} {'Buy&Hold':>12}")
print("-" * 42)
print(f"{'总收益率':<15} {lstm_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {lstm_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {lstm_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {lstm_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

plt.figure(figsize=(12, 5))
dates = pd.to_datetime(test_dates[:len(lstm_port)])
plt.plot(dates, lstm_port, label='LSTM',       linewidth=1.5, color='purple')
plt.plot(dates, bh_port,   label='Buy & Hold', linewidth=1.5, color='orange')
plt.title('Portfolio: LSTM vs Buy & Hold (2022-2024)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lstm_backtest.png', dpi=100, bbox_inches='tight')
plt.show()
print("回测图已保存！")