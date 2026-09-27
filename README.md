# 🪙 Bitcoin Multi-Horizon Price Prediction using Evolutionary Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, hardware-aware evolutionary machine learning system for forecasting Bitcoin (BTC/USDT) returns and directional movements simultaneously across multiple time horizons: **1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour, 4 hours, and 12 hours**.

---

## 🌟 Key Features

1. **High-Frequency 1-Year Dataset**:
   - Ingests 1 full year of 1-minute OHLCV candlestick bars (525,600+ records) directly from the Binance public data archive.
   - Computes 48 multi-timeframe technical, statistical, and cyclical features (RSI, MACD, Bollinger Bands, ATR/NATR, EMA distance/slopes, Taker Buy ratios, Volume dynamics).
   - Strict temporal splitting (70% train, 15% validation, 15% test) with leak-free scaling.

2. **Automated Hardware & NVLink Interconnect Detection**:
   - Automatically detects CPU physical/logical cores, CUDA-capable GPUs, and NVLink high-speed peer-to-peer topologies.
   - **NVLink Available**: Deploys unified Multi-GPU distributed training with high-bandwidth P2P tensor synchronization.
   - **NVLink Unavailable / Discrete GPUs / Multi-Core CPU**: Spawns parallel evolutionary worker instances evaluating distributed subpopulations with inter-generational tournament recombination.

3. **Multi-Horizon Residual Neural Architecture (`MultiHorizonBTCNet`)**:
   - Gated Feature Attention mechanism dynamically weighting input signals.
   - Deep Residual blocks with GELU activations and Layer Normalization.
   - 7 distinct parallel output heads predicting continuous log-returns and binary directional probabilities for $h \in [1\text{m}, 5\text{m}, 15\text{m}, 30\text{m}, 1\text{h}, 4\text{h}, 12\text{h}]$.

4. **Evolutionary Optimization & Resumable 5-Hour Training Engine**:
   - Population-based Genetic Evolution with Elitism, Tournament Selection, Arithmetic Crossover, and Adaptive Gaussian Mutation.
   - Configured with a default **5-hour continuous evolutionary budget** (`max_duration_seconds = 18000`).
   - Complete state persistence (`checkpoints/checkpoint_latest.pt` and `checkpoints/best_model.pt`) enabling uninterrupted training resumption after pause or process interruption.

5. **Diagnostic Backtesting & Analytics**:
   - Out-of-sample directional accuracy, simulated trading Sharpe ratios, win rates, maximum drawdowns, and comparison against Buy & Hold benchmark.

---

## 📂 Repository Structure

```tree
├── btc_price.ipynb                 # Comprehensive Jupyter Notebook (executable & pre-rendered)
├── system_detection.py             # Hardware, CUDA, and NVLink interconnect detection
├── download_data.py                # 1-Year 1-Minute Binance BTC/USDT ingestion pipeline
├── prepare_data.py                 # Feature engineering & multi-horizon target preparation
├── evolutionary_model.py           # MultiHorizonBTCNet architecture & genetic operators
├── evolutionary_trainer.py         # Resumable 5-hour evolutionary training engine
├── evaluate.py                     # Out-of-sample backtesting & performance visualization
├── data/
│   ├── btc_1m_1year.parquet        # 1-Year 1-Minute cleaned BTC dataset (525,600 bars)
│   ├── btc_1m_sample.csv           # Preview sample of raw OHLCV records
│   └── scaler_params.pkl           # Leak-free normalization parameters
├── checkpoints/
│   ├── best_model.pt               # Top-performing evolved model weights & config
│   ├── checkpoint_latest.pt        # Full resumable population & training state
│   └── training_metrics.json       # Generation-by-generation evolution metrics
├── plots/
│   ├── evaluation_metrics.png      # 4-panel diagnostic performance dashboard
│   └── test_evaluation_results.json# Detailed out-of-sample metrics per horizon
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/fgbvdcfgbfdefgb/btc-price-prediction.git
cd btc-price-prediction
pip install torch pandas numpy matplotlib psutil requests pyarrow
```

### 2. Check Hardware & NVLink

```bash
python system_detection.py
```

### 3. Ingest Data & Engineer Features

```bash
python download_data.py
python prepare_data.py
```

### 4. Run Resumable Evolutionary Training (5 Hours)

```bash
# Runs the evolutionary trainer for 5 hours (or resumes from previous checkpoint)
python evolutionary_trainer.py
```

To resume after an interruption:
```python
from evolutionary_trainer import EvolutionaryTrainer

trainer = EvolutionaryTrainer(max_duration_seconds=5 * 3600)
trainer.train(resume=True)
```

### 5. Evaluate Best Model & Generate Backtests

```bash
python evaluate.py
```

### 6. Interactive Jupyter Notebook

Launch the self-contained interactive notebook:
```bash
jupyter notebook btc_price.ipynb
```

---

## 📊 Backtest & Performance Across Horizons

| Horizon | Directional Accuracy | Win Rate | Simulated Sharpe | Horizon Focus |
|---|---|---|---|---|
| **1 Minute (1m)** | **51.3%** | 47.0% | Micro Scalping | High-frequency execution |
| **5 Minutes (5m)** | **50.9%** | 50.5% | Short Momentum | Intraday micro-trends |
| **15 Minutes (15m)** | **51.3%** | **51.2%** | **+1.05** | Breakout capture |
| **30 Minutes (30m)** | **50.9%** | 50.8% | Trend Filter | Multi-candle swing |
| **1 Hour (1h)** | **50.9%** | 50.9% | Session Trend | Hourly cycle trading |
| **4 Hours (4h)** | **51.3%** | 51.3% | Intermediate Trend | Macro intraday shifts |
| **12 Hours (12h)**| **49.7%** | 49.7% | Swing Bias | Multi-session directional bias |

---

## ⚙️ Evolutionary Algorithm Specifications

- **Chromosome Representation**:
  - Neural Network Weights ($W, b$) with Feature Attention Gate weights.
  - Architecture Hyperparameters: Hidden dimension ($64, 128, 192$), Depth ($2, 3, 4$), Dropout ($0.10 - 0.25$), Learning rate ($10^{-4} - 10^{-2}$).
- **Fitness Function**:
  $$\text{Fitness} = 100 \times \overline{\text{DA}} + 2 \times \overline{\text{Sharpe}} - 50 \times \text{MSE}$$
- **Genetic Recombination**: Arithmetic crossover with shape compatibility preservation.
- **Genetic Mutation**: Adaptive Gaussian noise $\mathcal{N}(0, \sigma^2)$ applied to active parameter tensors with mutation probability $p_m = 0.06$.
- **Selection Pressure**: Elitism retention (top $K=2$ or $4$) combined with Tournament Selection ($k=3$).

---

## 📜 License

MIT License. Free for academic, personal, and research use.
