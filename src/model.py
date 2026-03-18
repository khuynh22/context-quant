"""SignalNet — Lightweight binary Long/Short signal predictor.

Architecture:
  TemporalBranch : LSTM(seq_len, n_features → lstm_hidden) + attention pooling
  Head           : FC(lstm_hidden → fc_hidden → ReLU → Dropout → 1) → raw logit
  Output         : sigmoid(logit) = P(long) ∈ [0, 1]

Signal interpretation:
  signed_signal = 2 * P(long) - 1  ∈ [-1, 1]
  positive → BUY / long
  negative → SELL / put / short
  magnitude → position sizing proxy (|signed_signal| near 1 = high conviction)

Trained with return-weighted BCE (see train.py) so the model learns to be
right when the move is large — directly optimising expected PnL rather than
raw classification accuracy.

Usage
-----
>>> from src.model import SignalNet, build_model
>>> model = build_model()
>>> logit = model(x_temporal)                # [B]
>>> p_long = model.predict_long_prob(x_temporal)  # [B], no grad
"""
from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

INPUT_SIZE: Final[int] = 14    # 11 ticker features + 3 SPY market-regime features
LSTM_HIDDEN: Final[int] = 64   # kept small — prevents overfitting on ~3 K samples/ticker
FC_HIDDEN: Final[int] = 32
DROPOUT: Final[float] = 0.3


class SignalNet(nn.Module):
    """Binary Long/Short signal predictor optimised for PnL.

    Args:
        input_size  (int):   Features per time step (default 14).
        lstm_hidden (int):   LSTM hidden units (default 64).
        fc_hidden   (int):   FC hidden width (default 32).
        dropout     (float): Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        lstm_hidden: int = LSTM_HIDDEN,
        fc_hidden: int = FC_HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.attn_query = nn.Parameter(torch.randn(lstm_hidden))
        self.lstm_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Shape ``[B, seq_len, input_size]``.

        Returns:
            Raw logit ``[B]`` — positive bias toward long, negative toward short.
        """
        out, _ = self.lstm(x)  # [B, T, hidden]
        out = self.lstm_drop(out)
        scores = (out * self.attn_query).sum(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1)  # [B, T]
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden]
        return self.head(context).squeeze(-1)  # [B]

    def predict_long_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(long) ∈ [0, 1] without gradient.

        Args:
            x: Shape ``[B, seq_len, input_size]``.

        Returns:
            Tensor ``[B]`` of long probabilities.
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    input_size: int = INPUT_SIZE,
    lstm_hidden: int = LSTM_HIDDEN,
    fc_hidden: int = FC_HIDDEN,
    dropout: float = DROPOUT,
) -> SignalNet:
    """Instantiate a ``SignalNet`` with sensible defaults."""
    return SignalNet(
        input_size=input_size,
        lstm_hidden=lstm_hidden,
        fc_hidden=fc_hidden,
        dropout=dropout,
    )
