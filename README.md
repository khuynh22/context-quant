# ContextQuant: Multi-Modal Stock Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ContextQuant** is a deep learning-powered stock guidance system. Unlike traditional models that rely solely on historical price action, ContextQuant uses a **Multi-Modal Late Fusion** architecture to synthesize numerical technical indicators with real-time financial sentiment.

The goal is not to predict an exact "closing price," but to provide a **probabilistic guidance signal** (Strong Buy to Strong Sell) based on the convergence of market data and global news.

---

## ⚡ Getting Started (Environment Setup)

### 1. Install Miniconda
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Windows (Python 3.11 installer).

### 2. Create the environment
```bash
conda env create -f environment.yml
conda activate context-quant
```

### 3. Verify the setup
```bash
python verify_setup.py
```

### 4. Launch Jupyter
```bash
jupyter lab
```
Then open the `notebooks/` folder and start with `00_pytorch_fundamentals/`.

---

## 📚 Learning Path (Notebooks)

The `notebooks/` folder is structured to build skills progressively toward the full ContextQuant system.

| Module | Topics | Ties Into |
|--------|--------|-----------|
| `00_pytorch_fundamentals/` | Tensors, autograd, training loops | Everything |
| `01_time_series_basics/` | Pandas, NumPy, OHLCV data | Branch A (Temporal) |
| `02_neural_networks/` | Linear layers, activations, loss, optimizers | The Fusion Layer |
| `03_lstms_and_sequences/` | RNNs, LSTMs, sequence modeling | Branch A (Temporal) |
| `04_nlp_and_transformers/` | Tokenizers, BERT, FinBERT embeddings | Branch B (Linguistic) |
| `05_building_contextquant/` | Late Fusion Net, training, inference | The Full System |

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
* **Calculations:** `ta` for technical indicator generation (RSI, MACD, Bollinger Bands, etc.).
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
├── notebooks/
│   ├── 00_pytorch_fundamentals/   ← START HERE
│   ├── 01_time_series_basics/
│   ├── 02_neural_networks/
│   ├── 03_lstms_and_sequences/
│   ├── 04_nlp_and_transformers/
│   └── 05_building_contextquant/
├── src/
│   ├── data_loader.py  # Data ingestion & preprocessing
│   ├── model.py        # PyTorch architecture (ContextQuantFusionNet)
│   ├── train.py        # Training & validation loop
│   └── utils.py        # Technical indicators & helpers
├── data/               # Raw and processed CSVs (git-ignored)
├── models/             # Saved .pt model checkpoints (git-ignored)
├── app.py              # Streamlit Dashboard
├── environment.yml     # Conda environment (start here for setup)
└── verify_setup.py     # One-shot environment health check
