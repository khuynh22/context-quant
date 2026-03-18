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
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

WINDOW_SIZE: Final[int] = 60
FORWARD_DAYS: Final[int] = 5  # predict 5-day forward return (better SNR than 1-day)
CLASS_LABELS: Final[list[str]] = ["Sell", "Hold", "Buy"]
# Symmetric ±2% on 5-day return: cleaner signal than 5-class with ambiguous mid-boundaries
RETURN_THRESHOLDS: Final[tuple[float, float]] = (-0.02, 0.02)
FEATURE_COLS: Final[list[str]] = [
    # Per-ticker stationary features (11)
    "log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close", "log_vol_chg",
    "RSI_14", "MACD", "MACD_SIGNAL",
    "bb_position", "bb_width", "atr_pct",
    # Market regime features from SPY (3) — same across all tickers on a given day
    "spy_ret", "spy_rsi", "spy_bb_pos",
]
DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"

# ── Sector mapping ────────────────────────────────────────────────────────────
# 0=Technology  1=Financials  2=Healthcare  3=Energy  4=ConsumerDisc
# 5=ConsumerStaples  6=Industrials  7=CommServices  8=REIT+Utilities+Materials
SECTOR_MAP: Final[dict[str, int]] = {
    # Technology
    "AAPL": 0, "MSFT": 0, "NVDA": 0, "GOOGL": 0, "META": 0, "AMZN": 0, "TSLA": 0,
    "TSM": 0, "AVGO": 0, "ORCL": 0, "AMD": 0, "CRM": 0, "ADBE": 0, "NFLX": 0,
    "INTC": 0, "QCOM": 0, "NOW": 0, "INTU": 0, "MU": 0, "PLTR": 0, "SNOW": 0,
    "UBER": 0, "ARM": 0, "ANET": 0, "MRVL": 0, "DDOG": 0,
    # Financials
    "JPM": 1, "BAC": 1, "GS": 1, "V": 1, "MA": 1, "WFC": 1, "MS": 1,
    "AXP": 1, "BLK": 1, "C": 1, "COF": 1,
    # Healthcare
    "UNH": 2, "JNJ": 2, "LLY": 2, "ABBV": 2, "PFE": 2, "MRK": 2, "TMO": 2,
    "AMGN": 2, "BMY": 2, "GILD": 2, "DHR": 2, "CVS": 2,
    # Energy
    "XOM": 3, "CVX": 3, "SLB": 3, "COP": 3, "EOG": 3, "OXY": 3, "MPC": 3,
    # Consumer Discretionary
    "WMT": 4, "HD": 4, "MCD": 4, "COST": 4, "NKE": 4, "SBUX": 4, "TGT": 4,
    "LOW": 4, "TJX": 4, "BKNG": 4, "ABNB": 4,
    # Consumer Staples
    "PG": 5, "KO": 5, "PEP": 5, "PM": 5, "MO": 5, "CL": 5, "MDLZ": 5,
    # Industrials
    "CAT": 6, "BA": 6, "GE": 6, "HON": 6, "UPS": 6, "FDX": 6, "LMT": 6,
    "RTX": 6, "DE": 6, "ETN": 6, "EMR": 6,
    # Communication Services
    "VZ": 7, "T": 7, "CMCSA": 7, "DIS": 7, "CHTR": 7,
    # REIT + Utilities + Materials (grouped — smaller universe)
    "AMT": 8, "PLD": 8, "EQIX": 8, "PSA": 8, "O": 8,
    "NEE": 8, "DUK": 8, "SO": 8, "D": 8, "AEP": 8,
    "LIN": 8, "APD": 8, "NUE": 8, "FCX": 8, "GOLD": 8,
}
NUM_SECTORS: Final[int] = 9  # 0-8; unknown tickers fall back to 8


def download_ohlcv(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
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


# Module-level SPY cache: keyed by (start, end) so we only download once per run.
_spy_context_cache: dict[str, pd.DataFrame] = {}


def _get_spy_context(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """Return SPY-based market regime features aligned to *dates*.

    Features:
        spy_ret     Daily log-return of SPY (broad market direction).
        spy_rsi     SPY RSI-14 (overbought / oversold regime).
        spy_bb_pos  Where SPY sits within its Bollinger Band (mean-reversion signal).

    The result is forward-filled so ticker holidays that fall on SPY trading days
    don't introduce NaNs. Rows still missing after ffill (e.g. before SPY listing)
    are left as NaN and dropped by the caller's ``dropna()``.

    Args:
        dates: DatetimeIndex of the ticker's rows to align to.
        start: Data download start date.
        end: Data download end date.

    Returns:
        pd.DataFrame with columns ``spy_ret``, ``spy_rsi``, ``spy_bb_pos``,
        indexed to *dates*.
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
    # Reindex to ticker dates and forward-fill trading-day gaps (e.g. market holidays)
    return spy_ctx.reindex(dates).ffill()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stationary features: log-returns, RSI, MACD, BB_POSITION/WIDTH, ATR.

    Args:
        df (pd.DataFrame): OHLCV DataFrame (must include ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume`` columns).

    Returns:
        pd.DataFrame: Copy of *df* with all per-ticker FEATURE_COLS populated. Raw OHLCV
            columns are replaced by stationary log-return equivalents so the model never
            sees raw price levels (which vary wildly across tickers and time).
            SPY market regime columns are added separately in ``build_dataset``.
    """
    df = df.copy()
    close = df["Close"]
    prev_close = close.shift(1)

    # Log-returns: stationary, cross-ticker comparable, no price-level bias
    df["log_ret_open"] = np.log(df["Open"] / prev_close)
    df["log_ret_high"] = np.log(df["High"] / prev_close)
    df["log_ret_low"] = np.log(df["Low"] / prev_close)
    df["log_ret_close"] = np.log(close / prev_close)
    # Replace zero volumes with NaN before log to avoid -inf on halted-trading days
    vol = df["Volume"].replace(0, np.nan)
    df["log_vol_chg"] = np.log(vol / vol.shift(1))

    # RSI and MACD: already relative, keep as-is
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    # Bollinger Band derived features: where price sits in the band + band width
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    band_range = (bb_high - bb_low).replace(0, np.nan)
    df["bb_position"] = (close - bb_low) / band_range  # 0=at lower band, 1=at upper band
    df["bb_width"] = (bb_high - bb_low) / bb_mid  # normalized volatility

    # ATR as % of price: volatility independent of BB
    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    ).average_true_range()
    df["atr_pct"] = atr / close

    return df.dropna()


def create_5class_labels(df: pd.DataFrame) -> pd.Series:
    """Assign a 5-class directional label based on next-day return.

    Classes map to indices 0-2::

        0  Sell          5-day ret < -2 %
        1  Hold          -2 % ≤ ret < +2 %
        2  Buy           ret ≥ +2 %

    Args:
        df (pd.DataFrame): DataFrame that contains a ``Close`` column.

    Returns:
        pd.Series: Integer series aligned to *df* (last FORWARD_DAYS rows are NaN).
    """
    next_return = df["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    t = RETURN_THRESHOLDS
    labels = pd.cut(
        next_return,
        bins=[-np.inf, t[0], t[1], np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")
    return labels


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
    y_return: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Convert a flat 2-D feature array into overlapping sequences.

    Args:
        X (np.ndarray): Shape ``(T, F)``. Normalized feature matrix.
        y (np.ndarray): Shape ``(T,)``. Integer class labels.
        window_size (int): Number of time steps per input window (default 60).
        y_return (np.ndarray, optional): Shape ``(T,)``. Raw forward log-return targets
            for the regression head. Returned as a third element when provided.

    Returns:
        Tuple of ``(X_seq, y_seq)`` — or ``(X_seq, y_seq, y_ret_seq)`` when *y_return* given.
    """
    X_seq, y_seq = [], []
    y_ret_seq: list = [] if y_return is not None else []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size : i])
        y_seq.append(y[i])
        if y_return is not None:
            y_ret_seq.append(y_return[i])
    if y_return is not None:
        return (
            np.array(X_seq, dtype=np.float32),
            np.array(y_seq, dtype=np.int64),
            np.array(y_ret_seq, dtype=np.float32),
        )
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


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
        ticker (str): Yahoo Finance ticker symbol.
        start (str): Start date for data download (default ``"2010-01-01"``).
        end (str, optional): End date for data download. Defaults to today's date so the
            dataset is always current when re-run.
        window_size (int): Length of each input sequence in trading days (default 60).
        train_ratio (float): Fraction for train split; rest splits between val and test.
        val_ratio (float): Fraction for validation split.
        save (bool): If True, save ``.npz`` files to *processed_dir*.
        processed_dir (Path, optional): Destination folder for processed artefacts. Defaults to
            ``<repo_root>/data/processed/``.

    Returns:
        dict: Keys ``X_train``, ``y_train``, ``y_return_train``, ``X_val``, ``y_val``,
            ``y_return_val``, ``X_test``, ``y_test``, ``y_return_test``,
            ``sector_train``, ``sector_val``, ``sector_test`` as ``torch.Tensor`` on CPU.
            ``y_return_*`` are raw 5-day forward log-returns used by the regression head.
    """
    if processed_dir is None:
        processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = download_ohlcv(ticker, start=start, end=end)
    df = add_technical_indicators(df)

    # Join SPY market regime features (spy_ret, spy_rsi, spy_bb_pos)
    spy_ctx = _get_spy_context(df.index, start=start, end=end)
    df = df.join(spy_ctx, how="left")

    # Compute raw 5-day forward log return BEFORE masking so indices stay aligned.
    # This is the continuous regression target; create_5class_labels bins it into 0/1/2.
    forward_log_ret = np.log(df["Close"].shift(-FORWARD_DAYS) / df["Close"])

    # Drop the last FORWARD_DAYS rows — no forward return available for labelling.
    labels = create_5class_labels(df)
    valid_mask = labels.notna()
    df = df[valid_mask].copy()
    labels = labels[valid_mask]
    forward_log_ret = forward_log_ret[valid_mask]

    # Drop any remaining rows with NaN (e.g. SPY features missing at edges)
    feature_mask = df[FEATURE_COLS].notna().all(axis=1)
    df = df[feature_mask]
    labels = labels[feature_mask]
    forward_log_ret = forward_log_ret[feature_mask]

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = labels.to_numpy(dtype=np.int64)
    y_ret_all = forward_log_ret.to_numpy(dtype=np.float32)

    # Sector id: constant for this ticker, broadcast to every window
    sector_id = SECTOR_MAP.get(ticker.upper(), 8)

    # Chronological split — no shuffle to avoid look-ahead leakage.
    n = len(df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train_raw, y_train, y_ret_train = X_all[:n_train], y_all[:n_train], y_ret_all[:n_train]
    X_val_raw = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]
    y_ret_val = y_ret_all[n_train : n_train + n_val]
    X_test_raw = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]
    y_ret_test = y_ret_all[n_train + n_val :]

    # Fit scaler on train only to prevent leakage into val/test.
    # StandardScaler (z-score) is appropriate for log-returns which are ~normal.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    X_train_seq, y_train_seq, y_ret_train_seq = make_windows(
        X_train_sc, y_train, window_size, y_return=y_ret_train
    )
    X_val_seq, y_val_seq, y_ret_val_seq = make_windows(
        X_val_sc, y_val, window_size, y_return=y_ret_val
    )
    X_test_seq, y_test_seq, y_ret_test_seq = make_windows(
        X_test_sc, y_test, window_size, y_return=y_ret_test
    )

    # Broadcast sector_id to match number of windows per split
    sec_train = np.full(len(X_train_seq), sector_id, dtype=np.int64)
    sec_val = np.full(len(X_val_seq), sector_id, dtype=np.int64)
    sec_test = np.full(len(X_test_seq), sector_id, dtype=np.int64)

    log.info(
        "Shapes — train %s, val %s, test %s",
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape,
    )

    if save:
        out_path = processed_dir / f"{ticker}_temporal.npz"
        np.savez(
            out_path,
            X_train=X_train_seq, y_train=y_train_seq, y_return_train=y_ret_train_seq,
            X_val=X_val_seq, y_val=y_val_seq, y_return_val=y_ret_val_seq,
            X_test=X_test_seq, y_test=y_test_seq, y_return_test=y_ret_test_seq,
            sector_train=sec_train, sector_val=sec_val, sector_test=sec_test,
        )
        log.info("Saved processed tensors → %s", out_path)

    return {
        "X_train": torch.tensor(X_train_seq),
        "y_train": torch.tensor(y_train_seq),
        "y_return_train": torch.tensor(y_ret_train_seq),
        "X_val": torch.tensor(X_val_seq),
        "y_val": torch.tensor(y_val_seq),
        "y_return_val": torch.tensor(y_ret_val_seq),
        "X_test": torch.tensor(X_test_seq),
        "y_test": torch.tensor(y_test_seq),
        "y_return_test": torch.tensor(y_ret_test_seq),
        "sector_train": torch.tensor(sec_train),
        "sector_val": torch.tensor(sec_val),
        "sector_test": torch.tensor(sec_test),
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
    60-day windows are then concatenated across all tickers so
    the LSTM learns general price-movement patterns rather than one stock's
    behaviour. SPY market context is cached and reused across all tickers.

    Args:
        tickers (list[str]): Yahoo Finance ticker symbols, e.g. ``["AAPL", "MSFT", "NVDA"]``.
        start (str): Start date (default ``"2010-01-01"``).
        end (str, optional): End date; defaults to today so the dataset is always current.
        window_size (int): Lookback window in trading days (default 60).
        train_ratio (float): Fraction for train split.
        val_ratio (float): Fraction for validation split.

    Returns:
        dict: Same keys as :func:`build_dataset` but with samples from all tickers pooled together.
    """
    splits: dict[str, list[torch.Tensor]] = {
        "X_train": [], "y_train": [], "y_return_train": [],
        "X_val": [], "y_val": [], "y_return_val": [],
        "X_test": [], "y_test": [], "y_return_test": [],
        "sector_train": [], "sector_val": [], "sector_test": [],
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
        ticker (str): Ticker used when ``build_dataset`` was called.
        processed_dir (Path, optional): Folder containing the ``.npz`` file; defaults to
            ``<repo_root>/data/processed/``.

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
