"""Data ingestion & preprocessing pipeline.

Responsibilities:
- Download OHLCV data from yfinance
- Fetch financial headlines
- Compute technical indicators with the `ta` library
- Normalize and create sliding windows
- Save to data/processed/
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
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

WINDOW_SIZE: Final[int] = 60
CLASS_LABELS: Final[list[str]] = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
RETURN_THRESHOLDS: Final[tuple[float, float, float, float]] = (-0.02, -0.005, 0.005, 0.02)
FEATURE_COLS: Final[list[str]] = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD", "MACD_SIGNAL",
    "BB_MID", "BB_HIGH", "BB_LOW",
]
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"


def download_ohlcv(
    ticker: str,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance via yfinance.

    Args:
        ticker (str):Ticker symbol, e.g. ``"AAPL"``.
        start (str): ISO date string for the start of the period.
        end (str): ISO date string for the end of the period.

    Returns:
        pd.DataFrame: DatetimeIndex DataFrame with columns ``[Open, High, Low, Close, Volume]``.
            Falls back to synthetic data when the network is unavailable.
    """
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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI-14, MACD, and Bollinger Bands using the `ta` library.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must include a ``Close`` column).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with six additional columns:
        ``RSI_14``, ``MACD``, ``MACD_SIGNAL``, ``BB_MID``, ``BB_HIGH``, ``BB_LOW``.
    """
    df = df.copy()
    prices = df["Close"]

    df["RSI_14"] = ta.momentum.RSIIndicator(close=prices, window=14).rsi()

    macd = ta.trend.MACD(close=prices, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=prices, window=20, window_dev=2)
    df["BB_MID"] = bb.bollinger_mavg()
    df["BB_HIGH"] = bb.bollinger_hband()
    df["BB_LOW"] = bb.bollinger_lband()

    return df.dropna()


def create_5class_labels(df: pd.DataFrame) -> pd.Series:
    """Assign a 5-class directional label based on next-day return.

    Classes map to indices 0-4::

        0  Strong Sell   ret < -2 %
        1  Sell          -2 % ≤ ret < -0.5 %
        2  Hold          -0.5 % ≤ ret < +0.5 %
        3  Buy           +0.5 % ≤ ret < +2 %
        4  Strong Buy    ret ≥ +2 %

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains a ``Close`` column.

    Returns
    -------
    pd.Series
        Integer series aligned to *df* (last row is NaN-dropped).
    """
    next_return = df["Close"].pct_change(1).shift(-1)
    t = RETURN_THRESHOLDS
    labels = pd.cut(
        next_return,
        bins=[-np.inf, t[0], t[1], t[2], t[3], np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("Int64")
    return labels


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a flat 2-D feature array into overlapping sequences.

    Parameters
    ----------
    X : np.ndarray, shape ``(T, F)``
        Normalized feature matrix.
    y : np.ndarray, shape ``(T,)``
        Integer class labels.
    window_size : int
        Number of time steps per input window (default 60).

    Returns
    -------
    X_seq : np.ndarray, shape ``(T-window_size, window_size, F)``
    y_seq : np.ndarray, shape ``(T-window_size,)``
    """
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size : i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def build_dataset(
    ticker: str = "AAPL",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    save: bool = True,
    processed_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end data pipeline: download → indicators → labels → split → scale → windows.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date range for data download.
    window_size : int
        Length of each input sequence in trading days (default 60).
    train_ratio, val_ratio : float
        Fractions for train/validation splits; rest goes to test.
    save : bool
        If True, save ``.npz`` files to *processed_dir*.
    processed_dir : Path, optional
        Destination folder for processed artefacts.  Defaults to
        ``<repo_root>/data/processed/``.

    Returns
    -------
    dict with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
    ``X_test``, ``y_test`` as ``torch.Tensor`` on CPU.
    """
    if processed_dir is None:
        processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = download_ohlcv(ticker, start=start, end=end)
    df = add_technical_indicators(df)

    # Drop the last row — no next-day return for labelling.
    labels = create_5class_labels(df)
    df = df.iloc[:-1].copy()
    labels = labels.iloc[:-1].dropna()

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = labels.to_numpy(dtype=np.int64)

    # Chronological split — no shuffle to avoid look-ahead leakage.
    n = len(df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train_raw, y_train = X_all[:n_train], y_all[:n_train]
    X_val_raw, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    X_test_raw, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]

    # Fit scaler on train only to prevent leakage into val/test.
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    X_train_seq, y_train_seq = make_windows(X_train_sc, y_train, window_size)
    X_val_seq, y_val_seq = make_windows(X_val_sc, y_val, window_size)
    X_test_seq, y_test_seq = make_windows(X_test_sc, y_test, window_size)

    log.info(
        "Shapes — train %s, val %s, test %s",
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape,
    )

    if save:
        out_path = processed_dir / f"{ticker}_temporal.npz"
        np.savez(
            out_path,
            X_train=X_train_seq, y_train=y_train_seq,
            X_val=X_val_seq, y_val=y_val_seq,
            X_test=X_test_seq, y_test=y_test_seq,
        )
        log.info("Saved processed tensors → %s", out_path)

    return {
        "X_train": torch.tensor(X_train_seq),
        "y_train": torch.tensor(y_train_seq),
        "X_val": torch.tensor(X_val_seq),
        "y_val": torch.tensor(y_val_seq),
        "X_test": torch.tensor(X_test_seq),
        "y_test": torch.tensor(y_test_seq),
    }


def load_processed(
    ticker: str = "AAPL",
    processed_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load previously saved ``.npz`` artefacts back as torch tensors.

    Parameters
    ----------
    ticker : str
        Ticker used when ``build_dataset`` was called.
    processed_dir : Path, optional
        Folder containing the ``.npz`` file; defaults to
        ``<repo_root>/data/processed/``.

    Returns
    -------
    dict with the same keys as :func:`build_dataset`.
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
