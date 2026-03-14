# ContextQuant: Multi-Modal Stock Intelligence

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ContextQuant** is a deep learning system that produces a 5-class stock guidance signal — `[Strong Sell, Sell, Hold, Buy, Strong Buy]` — by fusing two distinct information streams:

- **Temporal branch:** a stacked LSTM over a 60-day window of OHLCV data + technical indicators (RSI, MACD, Bollinger Bands)
- **Linguistic branch:** a FinBERT sentiment vector derived from financial news headlines

Both branches are merged by a fully-connected fusion head trained end-to-end.

---

## Quickstart

### 1. Prerequisites

- Python 3.11 installed ([python.org](https://www.python.org/downloads/))
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (`pip install uv` or the standalone installer)
- Internet access for the initial data download and (optionally) FinBERT model weights

### 2. Create the environment and install dependencies

```bash
uv sync
```

> This reads `pyproject.toml` and `uv.lock`, creates a `.venv`, and installs every dependency at the exact pinned versions — including PyTorch CPU wheels from the official PyTorch index.
>
> To also install the Jupyter and dev extras:
> ```bash
> uv sync --extra jupyter --extra dev
> ```

### 3. Verify the setup

```bash
uv run python verify_setup.py
```

Every line should show a checkmark. If anything fails, run `uv sync` again or check the error message.

### 4. Train a model

```bash
uv run python -m src.train --ticker AAPL --epochs 60
```

This downloads AAPL price data, computes technical indicators, trains the fusion model, and saves the best checkpoint to `models/AAPL_best.pt`. Training takes a few minutes on CPU.

You can swap the ticker for any symbol supported by Yahoo Finance (e.g. `MSFT`, `NVDA`, `TSLA`).

### 5. Launch the dashboard

```bash
uv run streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

- Pick a **ticker** and **date range** in the sidebar
- Select the checkpoint you just trained from the **Checkpoint** dropdown
- Optionally paste a **news headline** to incorporate live sentiment
- Click **Analyse** — the model outputs a guidance signal with a full probability breakdown

---

## Training options

```bash
uv run python -m src.train --ticker AAPL         # defaults: 60 epochs, lr=1e-3, batch=64, patience=10
uv run python -m src.train --ticker MSFT --epochs 100 --lr 5e-4 --batch-size 128
```

The best checkpoint is saved to `models/<ticker>_best.pt` and automatically picked up by the dashboard.

---

## Python API

```python
from src.data_loader import build_dataset
from src.model import build_model
from src.train import train, evaluate_test

data   = build_dataset("AAPL")            # download, indicators, windowing, splits
model  = build_model()
history = train(model, data, n_epochs=60) # trains + saves best checkpoint

results = evaluate_test(model, data)
print(results["report"])                  # per-class precision / recall / F1
```

---

## Architecture

```
Branch A (Temporal)   [B, 60, 11]  ──► 2-layer LSTM ──────────────────────┐
                                                                            ├──► Fusion FC ──► [B, 5]
Branch B (Linguistic) [B, 3]       ──► 2-layer MLP (FinBERT probs) ────────┘
```

The sentiment input is a `[positive, negative, neutral]` probability vector from `ProsusAI/finbert`. When no headline is provided the model falls back to a uniform neutral prior.

---

## Project structure

```text
├── app.py                          # Streamlit dashboard
├── src/
│   ├── data_loader.py              # OHLCV download, indicators, windowing, splits
│   ├── model.py                    # ContextQuantFusionNet (LSTM + MLP + fusion head)
│   ├── train.py                    # Training loop, early stopping, checkpointing
│   └── utils.py                    # Indicator helpers, seed, device, metrics
├── data/                           # Raw / processed data (git-ignored)
├── models/                         # .pt checkpoints (git-ignored)
├── notebooks/                      # Progressive learning curriculum (see below)
├── pyproject.toml                  # dependencies + uv index config
├── uv.lock                         # exact pinned versions (commit this)
├── .python-version                 # Python 3.11 pin for uv
└── verify_setup.py
```

---

## Notebooks

A self-contained learning curriculum lives in [`notebooks/`](notebooks/README.md) — six progressive modules covering PyTorch fundamentals through the full ContextQuant system. Each module is standalone and requires no imports from `src/`.

```bash
uv sync --extra jupyter
uv run jupyter lab    # then open notebooks/ and start at 00_pytorch_fundamentals/
```
