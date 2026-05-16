# RAG_Decision-Transformer-stock-trading
# RAG-Enhanced Decision Transformer for Stock Trading

> A reinforcement learning framework that combines Retrieval-Augmented Generation (RAG), FinBERT sentiment analysis, and Decision Transformer for intelligent stock trading decisions.

---

## Overview

This project proposes **RAG-DT**, a novel offline reinforcement learning framework for stock trading. The model enhances the standard Decision Transformer by integrating:

- **FinBERT** sentiment analysis on financial news
- **FAISS-based RAG retrieval** over historical technical indicator vectors
- A unified 21-dimensional enhanced state representation

The framework is evaluated on **GOOG daily stock data (2022–2024)** and benchmarked against six baseline methods.

---

## Architecture

```
Stock Price Data (OHLCV)        Financial News
         │                            │
         ▼                            ▼
Multimodal Feature Extraction    RAG Retrieval
  ├── Technical Indicators       (FAISS, cosine similarity)
  │   (MA5, MA20, RSI, MACD)     K=3 similar scenarios
  └── FinBERT Sentiment          → 9-dim retrieval vector
      [P+, P−, P0]
         │                            │
         └──────────── concat ─────────┘
                          │
                          ▼
              Decision Transformer
          (Enhanced State: 21-dim)
          2 layers · 4 heads · hidden 64
                          │
                          ▼
               Trading Action
            Buy  /  Hold  /  Sell
```

---

## Experimental Results

All experiments use GOOG stock data. The training set covers 2010–2022; the test set covers 2022–2024.

| Method | Sharpe Ratio | Max Drawdown |
|---|---|---|
| Buy & Hold | 0.37 | — |
| MACD | -0.12 | — |
| DQN | 0.29 | — |
| PPO | — | — (failed to open positions) |
| LSTM | 0.69 | — |
| Decision Transformer (standard) | 0.86 | -26.78% |
| DT + FinBERT Sentiment | 0.87 | -14.99% |
| **RAG-DT (ours)** | **0.85** | **-14.99%** |

> The RAG module contributes primarily to **drawdown control** rather than absolute return. The comparable Sharpe ratio with reduced drawdown demonstrates the value of retrieval-augmented context in risk management.

---

## Project Structure

```
RAG_Decision-Transformer-stock-trading/
├── README.md
├── .gitignore
├── LICENSE
│
├── DT_simple.py              # Standard Decision Transformer
├── day10_finbert_dt.py       # DT + FinBERT sentiment
├── day10_real_news.py        # DT + real GOOG news data
├── day11_rag.py              # RAG-DT (main model)
├── day11_rag_v2.py           # RAG-DT improved version
│
├── fetch_goog_news.py        # GOOG news data retrieval
├── plot_results.py           # Results visualization
├── param_sensitivity.py      # Parameter sensitivity analysis
│
├── baseline/
│   ├── baseline_macd.py      # MACD strategy
│   ├── baseline_lstm.py      # LSTM baseline
│   └── baseline_dqn_ppo.py  # DQN and PPO baselines
│
└── data/                     # Data directory (not tracked)
    └── (goog_real_news.csv, goog_sentiment.csv)
```

---

## Environment Setup

```bash
# Create conda environment
conda create -n pytorch_env python=3.9
conda activate pytorch_env

# Install dependencies
pip install torch==2.8.0
pip install transformers
pip install faiss-cpu
pip install pandas numpy yfinance matplotlib
```

---

## Usage

**1. Fetch GOOG news data**
```bash
python fetch_goog_news.py
```

**2. Run baseline experiments**
```bash
python baseline/baseline_macd.py
python baseline/baseline_lstm.py
python baseline/baseline_dqn_ppo.py
```

**3. Train standard Decision Transformer**
```bash
python DT_simple.py
```

**4. Train DT with FinBERT sentiment**
```bash
python day10_real_news.py
```

**5. Train RAG-DT (main model)**
```bash
python day11_rag.py
```

**6. Visualize results**
```bash
python plot_results.py
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Sequence length | 20 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Hidden dimension | 64 |
| Learning rate | 1e-4 |
| Gradient clip | 0.3 |
| Training epochs | 15 |
| RAG neighbors (K) | 3 |
| State dimension | 21 |

---

## Paper

This project is the implementation of the paper:

> **RAG-Enhanced Decision Transformer for Stock Trading**
> Submitted to *IEEE Access / Neurocomputing* (EI-indexed)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
