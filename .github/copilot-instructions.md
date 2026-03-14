# Copilot Instructions — ContextQuant

## Project Overview

ContextQuant is a **multi-modal late-fusion** deep learning system for stock guidance. It combines a temporal branch (LSTM/TFT on 60-day OHLCV windows) with a linguistic branch (FinBERT sentiment) to produce a 5-class signal: `[Strong Sell, Sell, Hold, Buy, Strong Buy]`.

The repo has two surfaces: **teaching notebooks** (progressive curriculum in `notebooks/`) and a **production `src/` package** (stub files awaiting implementation per the phased roadmap).

## Architecture

```
Branch A (Temporal)  ──► LSTM/TFT on OHLCV + technical indicators ──┐
                                                                     ├──► Fusion FC layers ──► 5-class output
Branch B (Linguistic) ──► FinBERT on daily headline digest ──────────┘
```

Key libraries: PyTorch (CPU by default), Hugging Face `transformers` (`ProsusAI/finbert`), `yfinance`, `ta` (technical indicators), Streamlit (dashboard).

## Environment & Setup

- **uv env**: `uv sync` (reads `pyproject.toml` + `uv.lock`, creates `.venv` automatically)
- **Python**: 3.11 required (pinned in `.python-version`)
- **Verify**: `python verify_setup.py` — runs import checks and a tensor smoke test
- **Jupyter**: `jupyter lab` — notebooks expect the `context-quant` kernel
- PyTorch is CPU-only by default; change the `[tool.uv.sources]` index in `pyproject.toml` to a CUDA channel for GPU

## Project Layout

| Path | Purpose |
|------|---------|
| `notebooks/00–05_*/` | Progressive learning modules (00→05). Self-contained; do NOT import from `src/` |
| `src/data_loader.py` | Data ingestion: yfinance OHLCV, headline fetching, sliding windows |
| `src/model.py` | `ContextQuantFusionNet` PyTorch architecture (late fusion) |
| `src/train.py` | Training loop with early stopping, LR scheduling, checkpoint saving to `models/` |
| `src/utils.py` | Technical indicator helpers (RSI, MACD, Bollinger Bands via `ta` lib) |
| `data/` | Raw/processed CSVs (git-ignored) |
| `models/` | Saved `.pt` checkpoints (git-ignored) |

## Notebook Conventions

When creating or editing notebooks, follow these patterns observable across all existing notebooks:

- **Title cell**: `# NN — Title` as the first markdown cell, with a 1–2 line description
- **Sections**: `## Section N — Title` or `## N) Title`
- **Imports**: always the first code cell, ordered: stdlib → PyTorch → NumPy → other libs
- **Reproducibility**: call `torch.manual_seed(N)` right after imports when using randomness
- **Shape annotations**: comment tensor shapes inline, e.g. `# [batch, seq_len, features]`
- **Print everything**: every code cell ends with `print()` showing shapes, values, or status
- **Exercises**: every notebook ends with `## Exercises` — 2–3 numbered stretch tasks
- **Data paths**: use `pathlib.Path("../../data/")` (relative from notebook location)
- **Synthetic fallback**: generate fake data so notebooks run without real CSVs; guard real paths with `Path(...).exists()`

## Code Conventions

- **Naming**: `snake_case` for everything; `X` uppercase for feature tensors, `y` lowercase for labels
- **Batch vars**: `batch` or `xb`/`yb`; loaders are `train_loader`, `val_loader`, `test_loader`
- **Model var**: always stored as `model`
- **Device pattern**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **Comment banners**: use box-drawing chars for section separators — `# ── Section ────────`
- **Max line length**: 100 chars (flake8 config in `setup.cfg`)

## `src/` Implementation Guidelines

The `src/` modules are stubs with docstrings describing their responsibilities. When implementing:

- `data_loader.py` → use `yfinance` for OHLCV, `ta` library (not `pandas_ta`) for indicators, `MinMaxScaler` for normalization, 60-day sliding windows
- `model.py` → define `ContextQuantFusionNet(nn.Module)` with separate temporal and linguistic branches, fusion FC head, 5-class softmax output
- `train.py` → include early stopping, LR scheduling (`ReduceLROnPlateau`), save checkpoints to `models/` dir
- `utils.py` → wrap `ta` library calls for RSI, MACD, Bollinger Bands; keep functions pure and stateless
- Output class labels: `["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]`
