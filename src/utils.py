"""utils.py — Technical indicator helpers and shared utilities.

Pure, stateless functions used across data_loader, model, and train.

Sections
--------
1. Technical indicator wrappers (RSI, MACD, Bollinger Bands)
2. Training utilities  (seed, device, class weights)
3. Evaluation helpers  (accuracy, per-class report)
"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import ta
import torch

from sklearn.metrics import classification_report as skl_report
from sklearn.utils.class_weight import compute_class_weight

log = logging.getLogger(__name__)

CLASS_LABELS: Final[list[str]] = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

# ── 1. Technical indicators ──────────────────────────────────────────────────


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using the ``ta`` library.

    Args:
        prices (pd.Series): Closing prices.
        window (int): Look-back period (default 14).

    Returns:
        pd.Series: RSI values aligned to *prices*.
    """
    return ta.momentum.RSIIndicator(close=prices, window=window).rsi()


def compute_macd(
    prices: pd.Series,
    window_fast: int = 12,
    window_slow: int = 26,
    window_sign: int = 9,
) -> pd.DataFrame:
    """Compute MACD line and signal line.

    Args:
        prices (pd.Series): Closing prices.
        window_fast (int), window_slow (int), window_sign (int): EMA periods for the fast, slow,
            and signal smoothing.

    Returns:
        pd.DataFrame: Two-column DataFrame with ``MACD`` and ``MACD_SIGNAL``.
    """
    macd = ta.trend.MACD(
        close=prices,
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_sign,
    )
    return pd.DataFrame({"MACD": macd.macd(), "MACD_SIGNAL": macd.macd_signal()})


def compute_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    window_dev: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Args:
        prices (pd.Series): Closing prices.
        window (int): Rolling mean period (default 20).
        window_dev (float): Number of standard deviations for the band width (default 2).

    Returns:
        pd.DataFrame: Three-column DataFrame with ``BB_MID``, ``BB_HIGH``, ``BB_LOW``.
    """
    bb = ta.volatility.BollingerBands(close=prices, window=window, window_dev=window_dev)
    return pd.DataFrame(
        {
            "BB_MID": bb.bollinger_mavg(),
            "BB_HIGH": bb.bollinger_hband(),
            "BB_LOW": bb.bollinger_lband(),
        }
    )


# ── 2. Training utilities ────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training.

    Covers Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed (int): Seed value (default 42).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.debug("Seeds set to %d", seed)


def get_device() -> torch.device:
    """Return a CUDA device if available, otherwise CPU.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.debug("Using device: %s", device)
    return device


def compute_class_weights_tensor(
    y: np.ndarray | torch.Tensor,
    n_classes: int = 5,
) -> torch.Tensor:
    """Compute balanced class weights for a 5-class label vector.

    Useful for passing to ``nn.CrossEntropyLoss(weight=...)``.

    Args:
        y (array-like of int): Integer class labels (0-4).
        n_classes (int): Total number of classes (default 5).

    Returns:
        torch.Tensor: Shape (n_classes,), float weights inversely proportional to class frequency.
    """
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    y = y.astype(int)
    classes = np.arange(n_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


# ── 3. Evaluation helpers ────────────────────────────────────────────────────


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw logits or class predictions.

    Args:
        preds (torch.Tensor): Shape (N, C) or (N,). Model output logits or predicted class indices.
        labels (torch.Tensor): Shape (N,). Ground-truth class indices.

    Returns:
        float: Fraction of correct predictions in ``[0, 1]``.
    """
    if preds.dim() == 2:
        preds = preds.argmax(dim=1)
    return (preds == labels).float().mean().item()


def classification_report(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str] = CLASS_LABELS,
) -> str:
    """Return a human-readable per-class precision/recall/F1 table.

    Args:
        preds (torch.Tensor): Shape (N, C) or (N,). Model output logits or predicted class indices.
        labels (torch.Tensor): Shape (N,). Ground-truth class indices.
        class_names (list[str]): Labels for each class (default CLASS_LABELS).

    Returns:
        str: Multi-line report string.
    """
    if preds.dim() == 2:
        preds = preds.argmax(dim=1)
    y_pred = preds.cpu().numpy()
    y_true = labels.cpu().numpy()
    n = len(class_names)
    return skl_report(
        y_true, y_pred, labels=list(range(n)), target_names=class_names, zero_division=0
    )
