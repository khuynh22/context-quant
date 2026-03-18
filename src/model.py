"""ContextQuantFusionNet PyTorch architecture.

Architecture (Late Fusion, Dual-Head):
  Branch A (Temporal) : LSTM + attention on 60-day OHLCV+indicator+SPY windows [B, 60, 14]
  Branch B (Sector)   : Learnable embedding per market sector                   [B, 8]
  Branch C (Linguistic): MLP on FinBERT sentiment probabilities                 [B, 3]
  Shared Trunk         : Concat all → FC → ReLU → Dropout
  Head 1 (Class)       : FC → 3-class logits   (Sell / Hold / Buy)
  Head 2 (Return)      : FC → scalar           (predicted 5-day log return)

The dual-head design lets the model learn both *direction* (classification) and
*magnitude* (regression).  The regression output becomes the confidence proxy for
position sizing: a predicted return of +4% at P(Buy)=0.70 sizes larger than +1.5%
at P(Buy)=0.62.

Output classes (index 0-2): [Sell, Hold, Buy]
forward() returns: (class_logits [B, 3], return_pred [B])

Usage
-----
>>> from src.model import ContextQuantFusionNet, build_model
>>> model = build_model()
>>> logits, ret_pred = model(price_seq, sector_id)
>>> probs, ret_pred = model.predict_proba(price_seq, sector_id)
"""
from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

# Default hyper-parameters matching the data pipeline
TEMPORAL_INPUT_SIZE: Final[int] = 14   # 11 ticker features + 3 SPY market regime features
SENTIMENT_INPUT_SIZE: Final[int] = 3   # FinBERT: [positive, negative, neutral] probs
# Sectors: 0=Tech 1=Fin 2=Health 3=Energy 4=ConsDis 5=ConsStap 6=Indust 7=Comm 8=REIT+Util+Mat
NUM_SECTORS: Final[int] = 9
SECTOR_EMBED_DIM: Final[int] = 8       # small embedding; 9 sectors → 8-dim is plenty
LSTM_HIDDEN: Final[int] = 128
LSTM_LAYERS: Final[int] = 1            # keep 1 layer — 2 layers overfit too fast
FC_HIDDEN: Final[int] = 96
N_CLASSES: Final[int] = 3             # Sell / Hold / Buy
DROPOUT: Final[float] = 0.4


# ── Branch A: Temporal ───────────────────────────────────────────────────────

class TemporalBranch(nn.Module):
    """LSTM encoder with learned attention pooling over the full sequence."""

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
        self.attn_query = nn.Parameter(torch.randn(hidden_size))
        self.lstm_drop = nn.Dropout(0.2)
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.lstm_drop(out)
        scores = (out * self.attn_query).sum(dim=-1)
        weights = torch.softmax(scores, dim=-1)
        return (out * weights.unsqueeze(-1)).sum(dim=1)   # [B, hidden_size]


# ── Branch B: Linguistic ─────────────────────────────────────────────────────

class LinguisticBranch(nn.Module):
    """Two-layer MLP encoder for a FinBERT sentiment probability vector."""

    def __init__(
        self,
        input_size: int = SENTIMENT_INPUT_SIZE,
        hidden_size: int = FC_HIDDEN,
        dropout: float = 0.1,
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
        return self.net(x)


# ── Full Model (Dual-Head) ────────────────────────────────────────────────────

class ContextQuantFusionNet(nn.Module):
    """Multi-modal dual-head model: direction classification + return regression.

    The shared trunk fuses temporal, sector, and linguistic signals into a single
    latent representation. Two independent heads then read from it:

    - **ClassificationHead**: 3-class softmax (Sell / Hold / Buy)
    - **RegressionHead**: scalar 5-day log-return prediction

    The regression head's output is the primary confidence/sizing signal for the
    SignalEngine — a high P(Buy) with a predicted +4% return warrants a larger
    position than P(Buy)=0.62 with +1.2% predicted return.

    Args:
        temporal_input_size (int): Features per time step (default 14).
        sentiment_input_size (int): Sentiment vector dim (default 3).
        num_sectors (int): Number of distinct sector categories (default 9).
        sector_embed_dim (int): Sector embedding dimension (default 8).
        lstm_hidden (int): LSTM hidden size (default 128).
        lstm_layers (int): Number of stacked LSTM layers (default 1).
        fc_hidden (int): Shared trunk + linguistic branch hidden width (default 96).
        n_classes (int): Classification output classes (default 3).
        dropout (float): Dropout probability throughout (default 0.4).
    """

    def __init__(
        self,
        temporal_input_size: int = TEMPORAL_INPUT_SIZE,
        sentiment_input_size: int = SENTIMENT_INPUT_SIZE,
        num_sectors: int = NUM_SECTORS,
        sector_embed_dim: int = SECTOR_EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        fc_hidden: int = FC_HIDDEN,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        # ── Branches ─────────────────────────────────────────────────────────
        self.temporal_branch = TemporalBranch(
            input_size=temporal_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.sector_emb = nn.Embedding(num_sectors, sector_embed_dim)
        self.linguistic_branch = LinguisticBranch(
            input_size=sentiment_input_size,
            hidden_size=fc_hidden,
            dropout=dropout,
        )

        # ── Shared trunk ─────────────────────────────────────────────────────
        fused_dim = lstm_hidden + sector_embed_dim + fc_hidden
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),   # slightly less dropout on shared trunk
        )

        # ── Dual heads ───────────────────────────────────────────────────────
        self.cls_head = nn.Linear(fc_hidden, n_classes)
        self.ret_head = nn.Sequential(
            nn.Linear(fc_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._sentiment_input_size = sentiment_input_size
        self._num_sectors = num_sectors

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_sector: torch.Tensor | None = None,
        x_sentiment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_temporal: Shape ``[B, seq_len, temporal_input_size]``.
            x_sector: Shape ``[B]`` long tensor of sector IDs (0–8).
                Defaults to catch-all sector (8) when *None*.
            x_sentiment: Shape ``[B, sentiment_input_size]`` FinBERT probs.
                Defaults to neutral prior when *None*.

        Returns:
            tuple:
                class_logits (Tensor): Shape ``[B, n_classes]``, raw logits.
                return_pred  (Tensor): Shape ``[B]``, predicted 5-day log return.
        """
        B = x_temporal.size(0)
        device = x_temporal.device

        if x_sector is None:
            x_sector = torch.full(
                (B,), self._num_sectors - 1, dtype=torch.long, device=device
            )
        if x_sentiment is None:
            x_sentiment = torch.full(
                (B, self._sentiment_input_size),
                fill_value=1.0 / self._sentiment_input_size,
                dtype=x_temporal.dtype,
                device=device,
            )

        t_latent = self.temporal_branch(x_temporal)          # [B, lstm_hidden]
        s_latent = self.sector_emb(x_sector)                 # [B, sector_embed_dim]
        l_latent = self.linguistic_branch(x_sentiment)       # [B, fc_hidden]

        fused = torch.cat([t_latent, s_latent, l_latent], dim=1)  # [B, fused_dim]
        trunk_out = self.trunk(fused)                              # [B, fc_hidden]

        class_logits = self.cls_head(trunk_out)                    # [B, n_classes]
        return_pred = self.ret_head(trunk_out).squeeze(-1)         # [B]

        return class_logits, return_pred

    def predict_proba(
        self,
        x_temporal: torch.Tensor,
        x_sector: torch.Tensor | None = None,
        x_sentiment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return softmax class probabilities and predicted return.

        Returns:
            tuple:
                probs       (Tensor): Shape ``[B, n_classes]``, softmax probabilities.
                return_pred (Tensor): Shape ``[B]``, predicted 5-day log return.
        """
        with torch.no_grad():
            logits, ret = self.forward(x_temporal, x_sector, x_sentiment)
            return torch.softmax(logits, dim=1), ret

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(
    temporal_input_size: int = TEMPORAL_INPUT_SIZE,
    sentiment_input_size: int = SENTIMENT_INPUT_SIZE,
    num_sectors: int = NUM_SECTORS,
    sector_embed_dim: int = SECTOR_EMBED_DIM,
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
        num_sectors=num_sectors,
        sector_embed_dim=sector_embed_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        fc_hidden=fc_hidden,
        n_classes=n_classes,
        dropout=dropout,
    )
