import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== 生成图2：主实验结果对比图 ===")

# 加载所有保存的portfolio
data_path = '/Users/fengyihao/PyCharmMiscProject/深度学习/data/'

dates     = pd.to_datetime(np.load(data_path + 'test_dates.npy', allow_pickle=True))
bh_port   = np.load(data_path + 'bh_port.npy')
macd_port = np.load(data_path + 'macd_port.npy')
dqn_port  = np.load(data_path + 'dqn_port.npy')
lstm_port = np.load(data_path + 'lstm_port.npy')
dt_port   = np.load(data_path + 'dt_port.npy')
dt_sent_port = np.load(data_path + 'dt_sent_port.npy')
rag_port  = np.load(data_path + 'rag_port.npy')

print(f"数据加载成功！")
print(f"测试集天数：{len(dates)}")

# 统一长度（取最短的）
min_len = min(len(bh_port), len(macd_port), len(dqn_port),
              len(lstm_port), len(dt_port), len(rag_port))
dates_plot = dates[:min_len]

# 画图
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(dates[:len(bh_port)],      bh_port,      label='Buy & Hold',
        linewidth=1.5, color='gray',   linestyle='--')
ax.plot(dates[:len(macd_port)],    macd_port,    label='MACD',
        linewidth=1.5, color='brown',  linestyle='-.')
ax.plot(dates[:len(dqn_port)],     dqn_port,     label='DQN',
        linewidth=1.5, color='blue',   linestyle=':')
ax.plot(dates[:len(lstm_port)],    lstm_port,    label='LSTM',
        linewidth=1.5, color='purple', linestyle='-.')
ax.plot(dates[:len(dt_port)],      dt_port,      label='Standard DT',
        linewidth=1.5, color='orange', linestyle='-')
ax.plot(dates[:len(dt_sent_port)], dt_sent_port, label='DT+FinBERT',
        linewidth=2.0, color='green',  linestyle='-')
ax.plot(dates[:len(rag_port)],     rag_port,     label='RAG-DT (Ours)',
        linewidth=2.0, color='red',    linestyle='-')

ax.set_title('Portfolio Value Comparison (2022-2024)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value (USD)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/fengyihao/PyCharmMiscProject/深度学习/figure2_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("图2已保存！")

# ===== 图3：消融实验柱状图 =====
print("\n=== 生成图3：消融实验柱状图 ===")

methods = ['M1\n(RAG-DT)', 'M2\n(DT+Sentiment)', 'M3\n(Standard DT)', 'M4\n(LSTM)']
sharpe  = [0.85, 0.87, 0.86, 0.69]
maxdd   = [-14.99, -14.99, -26.78, -23.17]
colors  = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']

x     = np.arange(len(methods))
width = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 夏普比率
bars1 = ax1.bar(x, sharpe, width, color=colors,
                edgecolor='black', linewidth=0.5)
ax1.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
ax1.set_xlabel('Method', fontsize=11)
ax1.set_ylabel('Sharpe Ratio', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, sharpe):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

# 最大回撤
bars2 = ax2.bar(x, maxdd, width, color=colors,
                edgecolor='black', linewidth=0.5)
ax2.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
ax2.set_xlabel('Method', fontsize=11)
ax2.set_ylabel('Maximum Drawdown (%)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=10)
ax2.set_ylim(-35, 0)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, maxdd):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val - 0.5,
             f'{val:.2f}%', ha='center', va='top',
             fontsize=11, fontweight='bold')

plt.suptitle('Ablation Study Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/fengyihao/PyCharmMiscProject/深度学习/figure3_ablation.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("图3已保存！")