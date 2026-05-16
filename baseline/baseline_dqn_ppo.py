import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)

print("=== DQN & PPO Baseline ===")

# ===== 自定义交易环境 =====
class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df           = df.reset_index(drop=True)
        self.n_steps      = len(df)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None):
        self.current_step     = 0
        self.cash             = 10000
        self.shares           = 0
        self.portfolio_values = [10000]
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['Close_s'], row['Open_s'],   row['High_s'],
            row['Low_s'],   row['Volume_s'], row['MA5_s'],
            row['MA20_s'],  row['RSI_s'],    row['MACD_s']
        ], dtype=np.float32)

    def step(self, action):
        price = self.df.iloc[self.current_step]['Close']
        if action == 2 and self.cash > 0:
            self.shares, self.cash = self.cash / price, 0
        elif action == 0 and self.shares > 0:
            self.cash, self.shares = self.shares * price, 0
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        pv   = self.cash + self.shares * self.df.iloc[self.current_step]['Close']
        reward = (pv - self.portfolio_values[-1]) / self.portfolio_values[-1]
        self.portfolio_values.append(pv)
        return self._get_obs(), reward, done, False, {}

# ===== 准备数据 =====
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

# 按日期划分
train_stock = stock[stock['Date'] < '2022-01-01'].reset_index(drop=True)
test_stock  = stock[stock['Date'] >= '2022-01-01'].reset_index(drop=True)

print(f"训练集：{len(train_stock)}天，测试集：{len(test_stock)}天")

# 归一化（只用训练集拟合）
feature_cols = ['Close','Open','High','Low','Volume','MA5','MA20','RSI','MACD']
scaler       = MinMaxScaler()
scaled_train = scaler.fit_transform(train_stock[feature_cols])
scaled_test  = scaler.transform(test_stock[feature_cols])

for i, col in enumerate(feature_cols):
    train_stock[f'{col}_s'] = scaled_train[:, i]
    test_stock[f'{col}_s']  = scaled_test[:, i]

# ===== 训练DQN =====
print("\n训练DQN...")
dqn_env   = DummyVecEnv([lambda: StockTradingEnv(train_stock)])
dqn_model = DQN(
    "MlpPolicy", dqn_env,
    learning_rate=1e-4,
    buffer_size=10000,
    batch_size=64,
    seed=42,
    verbose=0
)
dqn_model.learn(total_timesteps=50000)
print("DQN训练完成！")

# ===== 训练PPO =====
print("\n训练PPO...")
ppo_env   = DummyVecEnv([lambda: StockTradingEnv(train_stock)])
ppo_model = PPO(
    "MlpPolicy", ppo_env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    seed=42,
    verbose=0
)
ppo_model.learn(total_timesteps=50000)
print("PPO训练完成！")

# ===== 回测函数 =====
def run_backtest(model, test_data):
    env = StockTradingEnv(test_data)
    obs, _ = env.reset()
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, done, _, _ = env.step(int(action))
    return actions

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

# 回测
prices      = test_stock['Close'].values
dqn_actions = run_backtest(dqn_model, test_stock)
ppo_actions = run_backtest(ppo_model, test_stock)

dqn_port = backtest(prices[:len(dqn_actions)], dqn_actions)
ppo_port = backtest(prices[:len(ppo_actions)], ppo_actions)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/dqn_port.npy', dqn_port)
np.save('/Users/fengyihao/PyCharmMiscProject/深度学习/data/ppo_port.npy', ppo_port)
bh_port  = backtest(prices, [2]+[1]*(len(prices)-1))

dqn_m = calc_metrics(dqn_port)
ppo_m = calc_metrics(ppo_port)
bh_m  = calc_metrics(bh_port)

print(f"\n{'指标':<15} {'DQN':>12} {'PPO':>12} {'Buy&Hold':>12}")
print("-" * 54)
print(f"{'总收益率':<15} {dqn_m[0]:>11.2f}% {ppo_m[0]:>11.2f}% {bh_m[0]:>11.2f}%")
print(f"{'年化收益率':<14} {dqn_m[1]:>11.2f}% {ppo_m[1]:>11.2f}% {bh_m[1]:>11.2f}%")
print(f"{'夏普比率':<15} {dqn_m[2]:>12.2f} {ppo_m[2]:>12.2f} {bh_m[2]:>12.2f}")
print(f"{'最大回撤':<15} {dqn_m[3]:>11.2f}% {ppo_m[3]:>11.2f}% {bh_m[3]:>11.2f}%")

print(f"\nDQN动作分布：卖出{dqn_actions.count(0)} 持有{dqn_actions.count(1)} 买入{dqn_actions.count(2)}")
print(f"PPO动作分布：卖出{ppo_actions.count(0)} 持有{ppo_actions.count(1)} 买入{ppo_actions.count(2)}")

# 可视化
plt.figure(figsize=(12, 5))
dates = pd.to_datetime(test_stock['Date'].values)
plt.plot(dates[:len(dqn_port)], dqn_port, label='DQN',        linewidth=1.5, color='blue')
plt.plot(dates[:len(ppo_port)], ppo_port, label='PPO',        linewidth=1.5, color='red')
plt.plot(dates[:len(bh_port)],  bh_port,  label='Buy & Hold', linewidth=1.5, color='orange')
plt.title('Portfolio: DQN & PPO vs Buy & Hold (2022-2024)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dqn_ppo_backtest.png', dpi=100, bbox_inches='tight')
plt.show()
print("回测图已保存！")