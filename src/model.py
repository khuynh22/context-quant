"""ContextQuantFusionNet PyTorch architecture.

Architecture (Late Fusion):
  Branch A (Temporal) : Stacked LSTM on 60-day OHLCV + indicator windows  [B, 60, 11]
  Branch B (Linguistic): MLP on FinBERT sentiment probabilities            [B, 3]
  Fusion Layer         : Concat → FC → Dropout → FC → 5-class logits      [B, 5]

Output classes (index 0-4): [Strong Sell, Sell, Hold, Buy, Strong Buy]

Usage
-----
>>> from src.model import ContextQuantFusionNet, build_model
>>> model = build_model()
>>> logits = model(price_seq, sentiment_vec)  # sentiment_vec optional (defaults to neutral)
"""
from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

# Default hyper-parameters matching the data pipeline
TEMPORAL_INPUT_SIZE: Final[int] = 11   # len(FEATURE_COLS) in data_loader
SENTIMENT_INPUT_SIZE: Final[int] = 3   # FinBERT: [positive, negative, neutral] probs
LSTM_HIDDEN: Final[int] = 128
LSTM_LAYERS: Final[int] = 2
FC_HIDDEN: Final[int] = 128
N_CLASSES: Final[int] = 5
DROPOUT: Final[float] = 0.3


# ── Branch A: Temporal ───────────────────────────────────────────────────────

class TemporalBranch(nn.Module):
    """Stacked LSTM encoder for an OHLCV + indicator sequence.

    Args:
        input_size (int): Number of features per time step (default 11).
        hidden_size (int): LSTM hidden dimension (default 128).
        num_layers (int): Number of stacked LSTM layers (default 2).
        dropout (float): Inter-layer dropout applied between LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_size: int = TEMPORAL_INPUT_SIZE,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): Shape ``[B, seq_len, input_size]``.

        Returns:
            torch.Tensor: Shape ``[B, hidden_size]``, last time-step hidden state.
        """
        out, _ = self.lstm(x)          # [B, seq_len, hidden_size]
        return out[:, -1, :]           # [B, hidden_size]


# ── Branch B: Linguistic ─────────────────────────────────────────────────────

class LinguisticBranch(nn.Module):
    """Two-layer MLP encoder for a FinBERT sentiment probability vector.

    Args:
        input_size (int): Dimension of the sentiment vector (default 3 = pos/neg/neu probs).
        hidden_size (int): Hidden layer width (default 128).
        dropout (float): Dropout applied before the second linear layer (default 0.3).
    """

    def __init__(
        self,
        input_size: int = SENTIMENT_INPUT_SIZE,
        hidden_size: int = FC_HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): Shape ``[B, input_size]``.

        Returns:
            torch.Tensor: Shape ``[B, hidden_size]``.
        """
        return self.net(x)             # [B, hidden_size]


# ── Fusion Head ───────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """FC fusion head that maps concatenated branch latents to 5-class logits.

    Args:
        temporal_dim (int): Output dimension of the temporal branch.
        linguistic_dim (int): Output dimension of the linguistic branch.
        hidden_size (int): Intermediate FC width (default 128).
        n_classes (int): Number of output classes (default 5).
        dropout (float): Dropout before the classification layer (default 0.3).
    """

    def __init__(
        self,
        temporal_dim: int = LSTM_HIDDEN,
        linguistic_dim: int = FC_HIDDEN,
        hidden_size: int = FC_HIDDEN,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(temporal_dim + linguistic_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(
        self, temporal_latent: torch.Tensor, linguistic_latent: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            temporal_latent (torch.Tensor): Shape ``[B, temporal_dim]``.
            linguistic_latent (torch.Tensor): Shape ``[B, linguistic_dim]``.

        Returns:
            torch.Tensor: Shape ``[B, n_classes]`` (raw logits).
        """
        fused = torch.cat([temporal_latent, linguistic_latent], dim=1)  # [B, T+L]
        return self.net(fused)                                          # [B, n_classes]


# ── Full Model ────────────────────────────────────────────────────────────────

class ContextQuantFusionNet(nn.Module):
    """Multi-modal late-fusion model for 5-class stock guidance.

    Combines a temporal LSTM branch (price/indicator windows) with a
    linguistic MLP branch (FinBERT sentiment) via a fusion FC head.

    Args:
        temporal_input_size (int): OHLCV + indicator features per time step (default 11).
        sentiment_input_size (int): Sentiment vector dimension (default 3).
        lstm_hidden (int): LSTM hidden size (default 128).
        lstm_layers (int): Number of stacked LSTM layers (default 2).
        fc_hidden (int): FC hidden size used in both branches and fusion (default 128).
        n_classes (int): Output classes (default 5).
        dropout (float): Dropout probability throughout (default 0.3).
    """

    def __init__(
        self,
        temporal_input_size: int = TEMPORAL_INPUT_SIZE,
        sentiment_input_size: int = SENTIMENT_INPUT_SIZE,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        fc_hidden: int = FC_HIDDEN,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.temporal_branch = TemporalBranch(
            input_size=temporal_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.linguistic_branch = LinguisticBranch(
            input_size=sentiment_input_size,
            hidden_size=fc_hidden,
            dropout=dropout,
        )
        self.fusion_head = FusionHead(
            temporal_dim=lstm_hidden,
            linguistic_dim=fc_hidden,
            hidden_size=fc_hidden,
            n_classes=n_classes,
            dropout=dropout,
        )
        self._sentiment_input_size = sentiment_input_size

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_sentiment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_temporal (torch.Tensor): Shape ``[B, seq_len, temporal_input_size]``.
                Normalised OHLCV + indicator windows.
            x_sentiment (torch.Tensor, optional): Shape ``[B, sentiment_input_size]``.
                FinBERT [positive, negative, neutral] probability vector.
                Defaults to a neutral prior ``[1/3, 1/3, 1/3]`` when *None*.

        Returns:
            torch.Tensor: Shape ``[B, n_classes]``, raw (un-softmaxed) logits.
        """
        if x_sentiment is None:
            B = x_temporal.size(0)
            x_sentiment = torch.full(
                (B, self._sentiment_input_size),
                fill_value=1.0 / self._sentiment_input_size,
                dtype=x_temporal.dtype,
                device=x_temporal.device,
            )

        t_latent = self.temporal_branch(x_temporal)       # [B, lstm_hidden]
        l_latent = self.linguistic_branch(x_sentiment)    # [B, fc_hidden]
        return self.fusion_head(t_latent, l_latent)        # [B, n_classes]

    def predict_proba(
        self,
        x_temporal: torch.Tensor,
        x_sentiment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return softmax class probabilities.

        Returns:
            torch.Tensor: Shape ``[B, n_classes]``, softmax class probabilities.
        """
        with torch.no_grad():
            return torch.softmax(self.forward(x_temporal, x_sentiment), dim=1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(
    temporal_input_size: int = TEMPORAL_INPUT_SIZE,
    sentiment_input_size: int = SENTIMENT_INPUT_SIZE,
    lstm_hidden: int = LSTM_HIDDEN,
    lstm_layers: int = LSTM_LAYERS,
    fc_hidden: int = FC_HIDDEN,
    n_classes: int = N_CLASSES,
    dropout: float = DROPOUT,
) -> ContextQuantFusionNet:
    """Instantiate a ``ContextQuantFusionNet`` with sensible defaults."""
    return ContextQuantFusionNet(
        temporal_input_size=temporal_input_size,
        sentiment_input_size=sentiment_input_size,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        fc_hidden=fc_hidden,
        n_classes=n_classes,
        dropout=dropout,
    )
