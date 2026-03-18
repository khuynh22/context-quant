"""signal_engine.py — Confidence-filtered trading signal generator.

Wraps a trained ContextQuantFusionNet and converts its dual-head output
(class probabilities + predicted return) into actionable trade signals with
ATR-based position sizing.

Signal logic
------------
1. Run predict_proba() → (probs [B,3], ret_pred [B]).
2. Classify as BUY (class 2) or SELL (class 0) only when the directional
   probability exceeds *conf_threshold* AND the predicted return sign is
   consistent with the direction.
3. Position size scales with excess confidence and shrinks with volatility:
       excess_conf  = (conf - threshold) / (1 - threshold)   → [0, 1]
       vol_factor   = target_vol / atr_pct                   (capped at 2)
       position_pct = min(max_position * excess_conf * vol_factor, max_position)
4. Stop-loss  = atr_pct * stop_atr_mult
   Take-profit = max(|ret_pred| * tp_ret_frac, stop_loss * min_rr_ratio)

Usage
-----
>>> from src.signal_engine import SignalEngine
>>> engine = SignalEngine.from_checkpoint("models/multi_90tickers_best.pt")
>>> signal = engine.generate(x_temporal, x_sector, current_atr_pct=0.012)
>>> print(signal)
{'action': 'BUY', 'confidence': 0.71, 'predicted_return': 0.038,
 'position_pct': 0.42, 'stop_loss_pct': 0.018, 'take_profit_pct': 0.030}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.model import ContextQuantFusionNet, build_model


@dataclass
class TradeSignal:
    """Output of :meth:`SignalEngine.generate`.

    Attributes:
        action: ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        confidence: Probability of the predicted directional class.
        predicted_return: Model's 5-day log-return estimate.
        position_pct: Suggested position size as a fraction of max capital (0–1).
        stop_loss_pct: Distance below (BUY) / above (SELL) entry to place stop (log-return).
        take_profit_pct: Distance in the signal direction to set target (log-return).
    """

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    predicted_return: float
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float

    def __str__(self) -> str:
        return (
            f"{self.action:4s} | conf={self.confidence:.2f} "
            f"ret_pred={self.predicted_return:+.3f} "
            f"size={self.position_pct:.2f} "
            f"SL={self.stop_loss_pct:.3f} TP={self.take_profit_pct:.3f}"
        )


class SignalEngine:
    """Wraps a trained dual-head model and produces ``TradeSignal`` objects.

    Args:
        model: Trained :class:`~src.model.ContextQuantFusionNet` in eval mode.
        conf_threshold: Minimum directional class probability to act (default 0.60).
        max_position: Maximum fraction of capital to allocate (default 1.0).
        target_vol: Daily ATR% that maps to a full-sized position (default 0.01 = 1%).
        stop_atr_mult: Stop-loss distance as a multiple of ATR (default 1.5).
        tp_ret_frac: Take-profit as a fraction of |predicted_return| (default 0.80).
        min_rr_ratio: Minimum reward-to-risk ratio; TP ≥ SL × this value (default 1.5).
    """

    def __init__(
        self,
        model: ContextQuantFusionNet,
        conf_threshold: float = 0.60,
        max_position: float = 1.0,
        target_vol: float = 0.01,
        stop_atr_mult: float = 1.5,
        tp_ret_frac: float = 0.80,
        min_rr_ratio: float = 1.5,
    ) -> None:
        self.model = model
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.max_position = max_position
        self.target_vol = target_vol
        self.stop_atr_mult = stop_atr_mult
        self.tp_ret_frac = tp_ret_frac
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
            checkpoint_path: Path to the checkpoint saved by :func:`~src.train.train`.
            device: Target device. Defaults to CPU.
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
        x_sector: torch.Tensor | None = None,
        x_sentiment: torch.Tensor | None = None,
        current_atr_pct: float = 0.01,
    ) -> TradeSignal:
        """Generate a :class:`TradeSignal` for a single asset.

        Args:
            x_temporal: Shape ``[1, seq_len, n_features]`` or ``[seq_len, n_features]``.
                The 60-day feature window for the asset.
            x_sector: Shape ``[1]`` long tensor of sector ID. Defaults to catch-all (8).
            x_sentiment: Shape ``[1, 3]`` FinBERT probs. Defaults to neutral prior.
            current_atr_pct: Current ATR as a fraction of price (e.g. 0.012 = 1.2%).
                Used to scale position size and compute stop-loss distance.

        Returns:
            :class:`TradeSignal` with action, confidence, position size, and levels.
        """
        if x_temporal.dim() == 2:
            x_temporal = x_temporal.unsqueeze(0)

        device = next(self.model.parameters()).device
        x_temporal = x_temporal.to(device)
        if x_sector is not None:
            x_sector = x_sector.to(device)
        if x_sentiment is not None:
            x_sentiment = x_sentiment.to(device)

        with torch.no_grad():
            probs, ret_pred = self.model.predict_proba(x_temporal, x_sector, x_sentiment)

        probs = probs[0]          # [3]
        ret_pred_val = ret_pred[0].item()

        sell_prob = probs[0].item()
        hold_prob = probs[1].item()   # noqa: F841 — kept for transparency
        buy_prob = probs[2].item()

        # Determine direction: only act if probability AND return sign agree
        if buy_prob >= self.conf_threshold and ret_pred_val > 0:
            action: Literal["BUY", "SELL", "HOLD"] = "BUY"
            confidence = buy_prob
        elif sell_prob >= self.conf_threshold and ret_pred_val < 0:
            action = "SELL"
            confidence = sell_prob
        else:
            return TradeSignal(
                action="HOLD",
                confidence=max(buy_prob, sell_prob),
                predicted_return=ret_pred_val,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
            )

        # Position sizing: scales with excess confidence, shrinks with volatility
        excess_conf = (confidence - self.conf_threshold) / (1.0 - self.conf_threshold)
        atr_pct = max(current_atr_pct, 1e-6)
        vol_factor = min(self.target_vol / atr_pct, 2.0)   # cap at 2× to avoid blowup
        position_pct = min(self.max_position * excess_conf * vol_factor, self.max_position)

        # Risk levels
        stop_loss_pct = atr_pct * self.stop_atr_mult
        tp_from_ret = abs(ret_pred_val) * self.tp_ret_frac
        tp_min = stop_loss_pct * self.min_rr_ratio
        take_profit_pct = max(tp_from_ret, tp_min)

        return TradeSignal(
            action=action,
            confidence=confidence,
            predicted_return=ret_pred_val,
            position_pct=round(position_pct, 4),
            stop_loss_pct=round(stop_loss_pct, 4),
            take_profit_pct=round(take_profit_pct, 4),
        )

    def generate_batch(
        self,
        x_temporal: torch.Tensor,
        x_sector: torch.Tensor | None = None,
        x_sentiment: torch.Tensor | None = None,
        atr_pcts: list[float] | None = None,
    ) -> list[TradeSignal]:
        """Generate signals for a batch of assets simultaneously.

        Args:
            x_temporal: Shape ``[B, seq_len, n_features]``.
            x_sector: Shape ``[B]`` sector IDs. Defaults to catch-all (8).
            x_sentiment: Shape ``[B, 3]``. Defaults to neutral prior.
            atr_pcts: Per-asset ATR fractions. If None, defaults to ``target_vol`` for all.

        Returns:
            List of :class:`TradeSignal` objects, one per item in the batch.
        """
        B = x_temporal.size(0)
        if atr_pcts is None:
            atr_pcts = [self.target_vol] * B

        device = next(self.model.parameters()).device
        x_temporal = x_temporal.to(device)
        if x_sector is not None:
            x_sector = x_sector.to(device)
        if x_sentiment is not None:
            x_sentiment = x_sentiment.to(device)

        with torch.no_grad():
            probs, ret_pred = self.model.predict_proba(x_temporal, x_sector, x_sentiment)

        signals = []
        for i in range(B):
            p = probs[i]
            r = ret_pred[i].item()
            sell_p, buy_p = p[0].item(), p[2].item()

            if buy_p >= self.conf_threshold and r > 0:
                action: Literal["BUY", "SELL", "HOLD"] = "BUY"
                conf = buy_p
            elif sell_p >= self.conf_threshold and r < 0:
                action = "SELL"
                conf = sell_p
            else:
                signals.append(TradeSignal("HOLD", max(buy_p, sell_p), r, 0.0, 0.0, 0.0))
                continue

            atr = max(atr_pcts[i], 1e-6)
            excess = (conf - self.conf_threshold) / (1.0 - self.conf_threshold)
            vol_f = min(self.target_vol / atr, 2.0)
            pos = min(self.max_position * excess * vol_f, self.max_position)
            sl = atr * self.stop_atr_mult
            tp = max(abs(r) * self.tp_ret_frac, sl * self.min_rr_ratio)
            signals.append(TradeSignal(action, conf, r, round(pos, 4), round(sl, 4), round(tp, 4)))

        return signals
