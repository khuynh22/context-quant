# ContextQuant: Multi-Modal Stock Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ContextQuant** is a deep learning-powered stock guidance system. Unlike traditional models that rely solely on historical price action, ContextQuant uses a **Multi-Modal Late Fusion** architecture to synthesize numerical technical indicators with real-time financial sentiment.

The goal is not to predict an exact "closing price," but to provide a **probabilistic guidance signal** (Strong Buy to Strong Sell) based on the convergence of market data and global news.

---

## 🚀 The Vision
Financial markets in 2026 are increasingly driven by "Narrative Violations." A company can have perfect technicals but crash on a single news headline. ContextQuant is built to:
1.  **Quantify Sentiment:** Use FinBERT to extract nuanced financial sentiment from news.
2.  **Analyze Momentum:** Use LSTMs/Transformers to identify technical patterns (RSI, MACD, Volume).
3.  **Fuse Modalities:** Combine these streams into a single decision-making "brain."

---

## 🧠 System Architecture

ContextQuant employs a **Late Fusion Network** implemented in PyTorch:

* **Branch A (Temporal):** A stacked LSTM or Temporal Fusion Transformer (TFT) processing a 60-day window of OHLCV data.
* **Branch B (Linguistic):** A pre-trained FinBERT encoder that processes a daily "Sentiment Digest" of headlines.
* **The Fusion Layer:** A series of fully connected layers that merge the latent representations from both branches to output a classification.

> **Output Schema:** `[Strong Sell, Sell, Hold, Buy, Strong Buy]` with associated confidence scores.

---

## 🛠️ Tech Stack
* **Framework:** PyTorch & PyTorch Lightning (for scalable training).
* **NLP:** Hugging Face `transformers` (specifically `ProsusAI/finbert`).
* **Data:** `yfinance` (Market Data) and `NewsAPI` or `AlphaVantage` (Financial News).
* **Calculations:** `pandas_ta` for technical indicator generation.
* **Dashboard:** Streamlit for the user-facing Guidance Report.

---

## 📊 Feature Roadmap
- [ ] **Phase 1: Data Engine**
  - Automated ingestion of ticker-specific price data.
  - Sentiment scraper for daily financial headlines.
- [ ] **Phase 2: Feature Engineering**
  - Normalization of time-series data using `MinMaxScaler`.
  - Tokenization and embedding of text data.
- [ ] **Phase 3: The Model**
  - Implementation of the ContextQuantFusionNet` in PyTorch.
  - Training loop with Early Stopping and Learning Rate scheduling.
- [ ] **Phase 4: Backtesting & Reliability**
  - A "Paper Trading" simulator to test historical accuracy.
  - Logic to handle "Slippage" and transaction fees.
- [ ] **Phase 5: Deployment**
  - Streamlit dashboard for real-time ticker analysis.

---

## 📂 Project Structure
```text
├── data/               # Raw and processed CSVs
├── models/             # Saved .pt model checkpoints
├── src/
│   ├── data_loader.py  # Data ingestion & preprocessing
│   ├── model.py        # PyTorch architecture
│   ├── train.py        # Training & Validation scripts
│   └── utils.py        # Technical indicators & helpers
├── app.py              # Streamlit Dashboard
└── requirements.txt
