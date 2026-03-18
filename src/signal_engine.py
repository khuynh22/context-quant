"""signal_engine.py — Binary Long/Short signal generator for SignalNet.

Wraps a trained SignalNet and converts its single sigmoid output (P(long))
into actionable trade signals with ATR-based position sizing.

Signal logic
------------
1. Run predict_long_prob(x_temporal) → P(long) ∈ [0, 1].
2. signed_signal = 2 * P(long) - 1  ∈ [-1, 1]
   - positive → BUY (long)
   - negative → SELL (short / put)
3. Only act when |signed_signal| > edge_threshold (default 0.20):
       P(long) > 0.60 → BUY     (model is >60 % confident in a long)
       P(long) < 0.40 → SELL    (model is >60 % confident in a short)
       else           → HOLD
4. Position size scales linearly with |signed_signal| and shrinks with volatility:
       edge       = |signed_signal| - edge_threshold   → [0, 1-threshold]
       edge_norm  = edge / (1 - edge_threshold)        → [0, 1]
       vol_factor = target_vol / atr_pct               (capped at 2)
       position_pct = min(max_position * edge_norm * vol_factor, max_position)
5. Stop-loss  = atr_pct * stop_atr_mult
   Take-profit = stop_loss * min_rr_ratio  (fixed R:R — no return prediction)

Usage
-----
>>> from src.signal_engine import SignalEngine
>>> engine = SignalEngine.from_checkpoint("models/AAPL_best.pt")
>>> signal = engine.generate(x_temporal, current_atr_pct=0.012)
>>> print(signal)
BUY  | conf=0.72 signal=+0.44 size=0.41 SL=0.018 TP=0.027
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.model import SignalNet, build_model


@dataclass
class TradeSignal:
    """Output of :meth:`SignalEngine.generate`.

    Attributes:
        action:          ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        confidence:      P(long) from the model (0–1).
        signed_signal:   2*P(long)-1; magnitude = conviction, sign = direction.
        position_pct:    Suggested position size as a fraction of max capital (0–1).
        stop_loss_pct:   Distance from entry to stop (log-return).
        take_profit_pct: Distance from entry to target (log-return).
    """

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    signed_signal: float
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float

    def __str__(self) -> str:
        return (
            f"{self.action:4s} | conf={self.confidence:.2f} "
            f"signal={self.signed_signal:+.3f} "
            f"size={self.position_pct:.2f} "
            f"SL={self.stop_loss_pct:.3f} TP={self.take_profit_pct:.3f}"
        )


class SignalEngine:
    """Wraps a trained ``SignalNet`` and produces :class:`TradeSignal` objects.

    Args:
        model           (SignalNet): Trained model in eval mode.
        edge_threshold  (float):     Minimum |signed_signal| to act (default 0.20).
                                     Equivalent to acting when P(long) > 0.60 or < 0.40.
        max_position    (float):     Maximum fraction of capital to allocate (default 1.0).
        target_vol      (float):     ATR% that maps to a full-sized position (default 0.01).
        stop_atr_mult   (float):     Stop-loss distance as a multiple of ATR (default 1.5).
        min_rr_ratio    (float):     Minimum reward-to-risk ratio (default 1.5).
    """

    def __init__(
        self,
        model: SignalNet,
        edge_threshold: float = 0.20,
        max_position: float = 1.0,
        target_vol: float = 0.01,
        stop_atr_mult: float = 1.5,
        min_rr_ratio: float = 1.5,
    ) -> None:
        self.model = model
        self.model.eval()
        self.edge_threshold = edge_threshold
        self.max_position = max_position
        self.target_vol = target_vol
        self.stop_atr_mult = stop_atr_mult
        self.min_rr_ratio = min_rr_ratio

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        **engine_kwargs,
    ) -> "SignalEngine":
        """Load a trained model from a ``.pt`` checkpoint file.

        Args:
            checkpoint_path: Path to a checkpoint saved by :func:`~src.train.train`.
            device:          Target device. Defaults to CPU.
            **engine_kwargs: Forwarded to :class:`SignalEngine.__init__`.

        Returns:
            :class:`SignalEngine` ready to call :meth:`generate`.
        """
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = build_model()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return cls(model, **engine_kwargs)

    # ── Core inference ────────────────────────────────────────────────────────

    def generate(
        self,
        x_temporal: torch.Tensor,
        current_atr_pct: float = 0.01,
    ) -> TradeSignal:
        """Generate a :class:`TradeSignal` for a single asset.

        Args:
            x_temporal:      Shape ``[1, seq_len, n_features]`` or ``[seq_len, n_features]``.
                             The 20-day feature window for the asset.
            current_atr_pct: Current ATR as a fraction of price (e.g. 0.012 = 1.2 %).
                             Used to scale position size and compute stop-loss distance.

        Returns:
            :class:`TradeSignal` with action, signed signal, position size, and levels.
        """
        if x_temporal.dim() == 2:
            x_temporal = x_temporal.unsqueeze(0)

        device = next(self.model.parameters()).device
        x_temporal = x_temporal.to(device)

        p_long = self.model.predict_long_prob(x_temporal)[0].item()
        signed = 2.0 * p_long - 1.0   # ∈ [-1, 1]
        edge = abs(signed)

        if edge <= self.edge_threshold:
            return TradeSignal(
                action="HOLD",
                confidence=p_long,
                signed_signal=signed,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
            )

        action: Literal["BUY", "SELL", "HOLD"] = "BUY" if signed > 0 else "SELL"

        edge_norm = (edge - self.edge_threshold) / (1.0 - self.edge_threshold)
        atr = max(current_atr_pct, 1e-6)
        vol_factor = min(self.target_vol / atr, 2.0)
        position_pct = min(self.max_position * edge_norm * vol_factor, self.max_position)

        stop_loss_pct = atr * self.stop_atr_mult
        take_profit_pct = stop_loss_pct * self.min_rr_ratio

        return TradeSignal(
            action=action,
            confidence=p_long,
            signed_signal=round(signed, 4),
            position_pct=round(position_pct, 4),
            stop_loss_pct=round(stop_loss_pct, 4),
            take_profit_pct=round(take_profit_pct, 4),
        )

    def generate_batch(
        self,
        x_temporal: torch.Tensor,
        atr_pcts: list[float] | None = None,
    ) -> list[TradeSignal]:
        """Generate signals for a batch of assets simultaneously.

        Args:
            x_temporal: Shape ``[B, seq_len, n_features]``.
            atr_pcts:   Per-asset ATR fractions. If None, defaults to ``target_vol`` for all.

        Returns:
            List of :class:`TradeSignal` objects, one per item in the batch.
        """
        B = x_temporal.size(0)
        if atr_pcts is None:
            atr_pcts = [self.target_vol] * B

        device = next(self.model.parameters()).device
        p_longs = self.model.predict_long_prob(x_temporal.to(device))

        signals = []
        for i in range(B):
            p_long = p_longs[i].item()
            signed = 2.0 * p_long - 1.0
            edge = abs(signed)
            atr = max(atr_pcts[i], 1e-6)

            if edge <= self.edge_threshold:
                signals.append(TradeSignal("HOLD", p_long, signed, 0.0, 0.0, 0.0))
                continue

            action: Literal["BUY", "SELL", "HOLD"] = "BUY" if signed > 0 else "SELL"
            edge_norm = (edge - self.edge_threshold) / (1.0 - self.edge_threshold)
            vol_f = min(self.target_vol / atr, 2.0)
            pos = min(self.max_position * edge_norm * vol_f, self.max_position)
            sl = atr * self.stop_atr_mult
            tp = sl * self.min_rr_ratio
            signals.append(
                TradeSignal(
                    action,
                    p_long,
                    round(signed, 4),
                    round(pos, 4),
                    round(sl, 4),
                    round(tp, 4)
                )
            )

        return signals
