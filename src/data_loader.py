"""Data ingestion & preprocessing pipeline.

Responsibilities:
- Download OHLCV data from yfinance
- Compute technical indicators with the `ta` library
- Create binary Long/Short labels from forward returns
- Normalise and create sliding windows
- Save to data/processed/

Label design
------------
Binary direction: 1 = long (forward return > 0), 0 = short (forward return ≤ 0).
No "Hold" class — the signal engine handles position sizing and confidence
thresholds so the model focuses purely on direction.

The regression target (raw 5-day forward log-return) is kept alongside the
binary label so train.py can weight each sample by |return|, making the loss
approximately proportional to expected PnL rather than raw accuracy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import ta
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

WINDOW_SIZE: Final[int] = 20   # 1 month of trading days — "small context"
FORWARD_DAYS: Final[int] = 5   # predict 5-day forward return
FEATURE_COLS: Final[list[str]] = [
    # Per-ticker stationary features (11)
    "log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close", "log_vol_chg",
    "RSI_14", "MACD", "MACD_SIGNAL",
    "bb_position", "bb_width", "atr_pct",
    # Market regime features from SPY (3) — same across all tickers on a given day
    "spy_ret", "spy_rsi", "spy_bb_pos",
]
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"


# ── OHLCV download ─────────────────────────────────────────────────────────────

def download_ohlcv(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance via yfinance.

    Args:
        ticker (str): Ticker symbol, e.g. ``"AAPL"``.
        start (str): ISO date string for the start of the period.
        end (str): ISO date string for the end of the period.

    Returns:
        pd.DataFrame: DatetimeIndex DataFrame with columns ``[Open, High, Low, Close, Volume]``.
            Falls back to synthetic data when the network is unavailable.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < WINDOW_SIZE + 30:
            raise ValueError(f"Downloaded only {len(df)} rows — too few.")
        log.info("Downloaded %d rows for %s", len(df), ticker)
    except Exception as exc:
        log.warning("yfinance download failed (%s); using synthetic fallback.", exc)
        df = _synthetic_ohlcv(n=1000)
    return df.sort_index()


def _synthetic_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate fake OHLCV so the pipeline runs without internet access."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.02, 1.2, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.6, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.3, n))
    volume = rng.integers(1_000_000, 6_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ── SPY market regime ──────────────────────────────────────────────────────────

_spy_context_cache: dict[str, pd.DataFrame] = {}


def _get_spy_context(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """Return SPY-based market regime features aligned to *dates*.

    Features:
        spy_ret     Daily log-return of SPY.
        spy_rsi     SPY RSI-14.
        spy_bb_pos  Where SPY sits within its Bollinger Band.

    Args:
        dates: DatetimeIndex of the ticker's rows to align to.
        start: Data download start date.
        end:   Data download end date.

    Returns:
        pd.DataFrame with columns ``spy_ret``, ``spy_rsi``, ``spy_bb_pos``.
    """
    cache_key = f"{start}_{end}"
    if cache_key not in _spy_context_cache:
        spy = download_ohlcv("SPY", start=start, end=end)
        close = spy["Close"]
        ctx = pd.DataFrame(index=spy.index)
        ctx["spy_ret"] = np.log(close / close.shift(1))
        ctx["spy_rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        band = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
        ctx["spy_bb_pos"] = (close - bb.bollinger_lband()) / band
        _spy_context_cache[cache_key] = ctx.dropna()
    spy_ctx = _spy_context_cache[cache_key]
    return spy_ctx.reindex(dates).ffill()


# ── Technical indicators ───────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stationary features: log-returns, RSI, MACD, BB, ATR.

    Args:
        df (pd.DataFrame): OHLCV DataFrame.

    Returns:
        pd.DataFrame: Copy of *df* with all per-ticker FEATURE_COLS populated.
    """
    df = df.copy()
    close = df["Close"]
    prev_close = close.shift(1)

    df["log_ret_open"] = np.log(df["Open"] / prev_close)
    df["log_ret_high"] = np.log(df["High"] / prev_close)
    df["log_ret_low"] = np.log(df["Low"] / prev_close)
    df["log_ret_close"] = np.log(close / prev_close)
    vol = df["Volume"].replace(0, np.nan)
    df["log_vol_chg"] = np.log(vol / vol.shift(1))

    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    band_range = (bb_high - bb_low).replace(0, np.nan)
    df["bb_position"] = (close - bb_low) / band_range
    df["bb_width"] = (bb_high - bb_low) / bb_mid

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    ).average_true_range()
    df["atr_pct"] = atr / close

    return df.dropna()


# ── Labels ────────────────────────────────────────────────────────────────────

def create_binary_labels(df: pd.DataFrame) -> pd.Series:
    """Assign a binary direction label based on 5-day forward return.

    Classes::

        1  Long    5-day forward return > 0
        0  Short   5-day forward return ≤ 0

    Args:
        df (pd.DataFrame): DataFrame with a ``Close`` column.

    Returns:
        pd.Series: Integer series (Int64) aligned to *df*. Last FORWARD_DAYS rows are NaN.
    """
    next_return = df["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    # pandas evaluates (NaN > 0) as False, not NA — explicitly restore NA so the
    # caller's notna() mask correctly drops the last FORWARD_DAYS rows.
    labels = (next_return > 0).astype("Int64")
    labels[next_return.isna()] = pd.NA
    return labels


# ── Windowing ─────────────────────────────────────────────────────────────────

def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
    y_return: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Convert a flat 2-D feature array into overlapping sequences.

    Args:
        X (np.ndarray):          Shape ``(T, F)``. Normalised feature matrix.
        y (np.ndarray):          Shape ``(T,)``. Binary labels (0/1).
        window_size (int):       Number of time steps per input window (default 20).
        y_return (np.ndarray):   Shape ``(T,)``. Raw forward log-return targets
                                 for loss weighting. Returned as third element when given.

    Returns:
        ``(X_seq, y_seq)`` or ``(X_seq, y_seq, y_ret_seq)`` when *y_return* given.
    """
    X_seq, y_seq, y_ret_seq = [], [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size : i])
        y_seq.append(y[i])
        if y_return is not None:
            y_ret_seq.append(y_return[i])
    X_out = np.array(X_seq, dtype=np.float32)
    y_out = np.array(y_seq, dtype=np.int64)
    if y_return is not None:
        return X_out, y_out, np.array(y_ret_seq, dtype=np.float32)
    return X_out, y_out


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    ticker: str = "AAPL",
    start: str = "2010-01-01",
    end: str | None = None,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    save: bool = True,
    processed_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end data pipeline: download → indicators → labels → split → scale → windows.

    Args:
        ticker (str):          Yahoo Finance ticker symbol.
        start (str):           Start date for data download (default ``"2010-01-01"``).
        end (str, optional):   End date. Defaults to today so the dataset is always current.
        window_size (int):     Length of each input sequence in trading days (default 20).
        train_ratio (float):   Fraction for train split.
        val_ratio (float):     Fraction for validation split.
        save (bool):           If True, save ``.npz`` to *processed_dir*.
        processed_dir (Path):  Destination folder. Defaults to ``<repo>/data/processed/``.

    Returns:
        dict with keys:
            ``X_train``, ``y_train``, ``y_return_train``,
            ``X_val``,   ``y_val``,   ``y_return_val``,
            ``X_test``,  ``y_test``,  ``y_return_test``  — all ``torch.Tensor`` on CPU.

        ``y_*`` are binary (0=short, 1=long).
        ``y_return_*`` are raw 5-day forward log-returns for loss weighting.
    """
    if processed_dir is None:
        processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = download_ohlcv(ticker, start=start, end=end)
    df = add_technical_indicators(df)

    spy_ctx = _get_spy_context(df.index, start=start, end=end)
    df = df.join(spy_ctx, how="left")

    # Compute raw 5-day forward log-return BEFORE masking (keeps index alignment).
    forward_log_ret = np.log(df["Close"].shift(-FORWARD_DAYS) / df["Close"])

    labels = create_binary_labels(df)
    valid_mask = labels.notna()
    df = df[valid_mask].copy()
    labels = labels[valid_mask]
    forward_log_ret = forward_log_ret[valid_mask]

    feature_mask = df[FEATURE_COLS].notna().all(axis=1)
    df = df[feature_mask]
    labels = labels[feature_mask]
    forward_log_ret = forward_log_ret[feature_mask]

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = labels.to_numpy(dtype=np.int64)
    y_ret = forward_log_ret.to_numpy(dtype=np.float32)

    n = len(df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train_raw = X_all[:n_train]
    y_train = y_all[:n_train]
    yr_train = y_ret[:n_train]
    X_val_raw = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]
    yr_val = y_ret[n_train : n_train + n_val]
    X_test_raw = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]
    yr_test = y_ret[n_train + n_val :]

    # Fit scaler on train only — no leakage into val/test.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    X_train_seq, y_train_seq, yr_train_seq = make_windows(
        X_train_sc, y_train, window_size, yr_train
    )
    X_val_seq, y_val_seq, yr_val_seq = make_windows(
        X_val_sc, y_val, window_size, yr_val
    )
    X_test_seq, y_test_seq, yr_test_seq = make_windows(
        X_test_sc, y_test, window_size, yr_test
    )

    log.info(
        "Shapes — train %s  val %s  test %s",
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape,
    )

    if save:
        out_path = processed_dir / f"{ticker}_temporal.npz"
        np.savez(
            out_path,
            X_train=X_train_seq, y_train=y_train_seq, y_return_train=yr_train_seq,
            X_val=X_val_seq, y_val=y_val_seq, y_return_val=yr_val_seq,
            X_test=X_test_seq, y_test=y_test_seq, y_return_test=yr_test_seq,
        )
        log.info("Saved processed tensors → %s", out_path)

    return {
        "X_train": torch.tensor(X_train_seq),
        "y_train": torch.tensor(y_train_seq),
        "y_return_train": torch.tensor(yr_train_seq),
        "X_val": torch.tensor(X_val_seq),
        "y_val": torch.tensor(y_val_seq),
        "y_return_val": torch.tensor(yr_val_seq),
        "X_test": torch.tensor(X_test_seq),
        "y_test": torch.tensor(y_test_seq),
        "y_return_test": torch.tensor(yr_test_seq),
    }


def build_multi_dataset(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Build one combined dataset from multiple tickers for multi-stock training.

    Each ticker is downloaded and scaled **independently** with its own
    ``StandardScaler`` fitted on that ticker's training split only. The
    20-day windows are then concatenated across all tickers so the LSTM
    learns general price-movement patterns. SPY market context is cached
    and reused across all tickers.

    Args:
        tickers (list[str]):   Yahoo Finance ticker symbols.
        start (str):           Start date (default ``"2010-01-01"``).
        end (str, optional):   End date; defaults to today.
        window_size (int):     Lookback window in trading days (default 20).
        train_ratio (float):   Fraction for train split.
        val_ratio (float):     Fraction for validation split.

    Returns:
        dict: Same keys as :func:`build_dataset` but with samples from all tickers pooled.
    """
    splits: dict[str, list[torch.Tensor]] = {
        "X_train": [], "y_train": [], "y_return_train": [],
        "X_val": [], "y_val": [], "y_return_val": [],
        "X_test": [], "y_test": [], "y_return_test": [],
    }

    for ticker in tickers:
        try:
            data = build_dataset(
                ticker,
                start=start,
                end=end,
                window_size=window_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                save=False,
            )
            for k in splits:
                splits[k].append(data[k])
            log.info(
                "Added %-6s — train=%d  val=%d  test=%d windows",
                ticker,
                len(data["X_train"]),
                len(data["X_val"]),
                len(data["X_test"]),
            )
        except Exception as exc:
            log.warning("Skipping %s: %s", ticker, exc)

    if not splits["X_train"]:
        raise RuntimeError("No ticker data was successfully downloaded.")

    return {k: torch.cat(v) for k, v in splits.items()}


def load_processed(
    ticker: str = "AAPL",
    processed_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load previously saved ``.npz`` artefacts back as torch tensors.

    Args:
        ticker (str):           Ticker used when ``build_dataset`` was called.
        processed_dir (Path):   Folder containing the ``.npz`` file.

    Returns:
        dict: Same keys as :func:`build_dataset`.
    """
    if processed_dir is None:
        processed_dir = DATA_DIR / "processed"
    path = processed_dir / f"{ticker}_temporal.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed data at {path}. Run build_dataset('{ticker}') first."
        )
    data = np.load(path)
    return {k: torch.tensor(data[k]) for k in data.files}
