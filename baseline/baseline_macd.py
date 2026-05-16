import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== MACD Baseline ===")

# 加载数据
df = pd.read_csv('/Users/fengyihao/PyCharmMiscProject/深度学习/data/selected_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
stock = df[df['Symbol'] == 'GOOG'].copy().reset_index(drop=True)

# 计算技术指标
stock['MA5']  = stock['Close'].rolling(5).mean()
stock['MA20'] = stock['Close'].rolling(20).mean()
ema12         = stock['Close'].ewm(span=12).mean()
ema26         = stock['Close'].ewm(span=26).mean()
stock['MACD'] = ema12 - ema26
stock['Signal'] = stock['MACD'].ewm(span=9).mean()  # 信号线
stock = stock.dropna().reset_index(drop=True)

# 只取测试集（2022-2024）
test_stock = stock[(stock['Date'] >= '2022-01-01') & (stock['Date'] <= '2024-12-31')].reset_index(drop=True)
print(f"测试集：{len(test_stock)}天")
print(f"时间范围：{test_stock['Date'].min().date()} 到 {test_stock['Date'].max().date()}")

# ===== MACD策略 =====
# MACD线上穿信号线 → 买入
# MACD线下穿信号线 → 卖出
actions = []
for i in range(len(test_stock)):
    if i < 1:
        actions.append(1)  # 第一天持有
        continue

    macd_prev   = test_stock['MACD'].iloc[i-1]
    signal_prev = test_stock['Signal'].iloc[i-1]
    macd_curr   = test_stock['MACD'].iloc[i]
    signal_curr = test_stock['Signal'].iloc[i]

    # 金叉：MACD从下方穿越信号线 → 买入
    if macd_prev < signal_prev and macd_curr >= signal_curr:
        actions.append(2)
    # 死叉：MACD从上方穿越信号线 → 卖出
    elif macd_prev > signal_prev and macd_curr <= signal_curr:
        actions.append(0)
    else:
        actions.append(1)  # 持有

test_stock['Action'] = actions

print(f"\n动作分布：")
print(f"  卖出(0)：{(test_stock['Action']==0).sum()}次")
print(f"  持有(1)：{(test_stock['Action']==1).sum()}次")
print(f"  买入(2)：{(test_stock['Action']==2).sum()}次")

# ===== 回测 =====
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

prices       = test_stock['Close'].values
macd_port    = backtest(prices, test_stock['Action'].values)
bh_port      = backtest(prices, np.array([2]+[1]*(len(prices)-1)))

macd_m = calc_metrics(macd_port)
bh_m   = calc_metrics(bh_port)

print(f"\n{'指标':<15} {'MACD策略':>12} {'Buy&Hold':>12}")
print("-" * 42)
print(f"{'总收益率':<15} {macd_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {macd_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {macd_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {macd_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

# 可视化
plt.figure(figsize=(12, 5))
dates = pd.to_datetime(test_stock['Date'].values)
plt.plot(dates, macd_port, label='MACD Strategy', linewidth=1.5, color='green')
plt.plot(dates, bh_port,   label='Buy & Hold',    linewidth=1.5, color='orange')
plt.title('Portfolio: MACD Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('macd_backtest.png', dpi=100, bbox_inches='tight')
plt.show()
print("回测图已保存！")