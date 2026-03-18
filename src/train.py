"""
train.py — Training loop for SignalNet, optimised for PnL.

Loss: Return-weighted Binary Cross-Entropy
    Each sample is weighted by |forward_return| so the model pays more
    attention to large moves (where being wrong is expensive) and ignores
    near-flat days (where direction barely matters).

    loss = mean( |ret| / mean(|ret|) * BCE(logit, label) )

    This is a differentiable proxy for expected PnL: a model that is right
    on big up-days and right on big down-days maximises this objective.

Primary metric: annualised Sharpe on the validation set.
    signal   = 2 * sigmoid(logit) - 1   ∈ [-1, 1]
    daily_pnl = signal * forward_return
    Sharpe    = mean(daily_pnl) / std(daily_pnl) * sqrt(252 / FORWARD_DAYS)

    Checkpoints are saved when val_sharpe improves, not when val_loss improves.

Typical usage
-------------
>>> from src.data_loader import build_dataset
>>> from src.model import build_model
>>> from src.train import train
>>> data = build_dataset("AAPL")
>>> model = build_model()
>>> history = train(model, data, n_epochs=80)
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.model import SignalNet, build_model
from src.utils import get_device, set_seed

log = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).parent.parent / "models"
FORWARD_DAYS: int = 5   # must match data_loader.FORWARD_DAYS


# ── Loss & metric ─────────────────────────────────────────────────────────────

def pnl_weighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Return-weighted BCE loss — approximately maximises expected PnL.

    Each sample's BCE contribution is scaled by |forward_return| (normalised
    to unit mean), so errors on large moves dominate the gradient.

    Args:
        logits  (Tensor): Raw model output ``[B]``.
        labels  (Tensor): Binary targets ``[B]`` (1=long, 0=short).
        returns (Tensor): 5-day forward log-returns ``[B]``.

    Returns:
        Tensor: Scalar loss.
    """
    weights = returns.abs()
    weights = weights / (weights.mean() + 1e-8)
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return (bce * weights).mean()


def compute_sharpe(
    logits: torch.Tensor,
    returns: torch.Tensor,
    annualize: bool = True,
) -> float:
    """Simulate trading PnL from model logits and return annualised Sharpe.

    signal = tanh(logit/2) approximates 2*sigmoid(logit)-1 smoothly in [-1,1].
    pnl    = signal * actual_return

    Args:
        logits   (Tensor): Raw model output ``[B]``.
        returns  (Tensor): 5-day forward log-returns ``[B]``.
        annualize (bool):  If True, scale to annual Sharpe (default True).

    Returns:
        float: Sharpe ratio (annualised when *annualize* is True).
    """
    with torch.no_grad():
        signal = torch.tanh(logits / 2.0)
        pnl = signal * returns
        sharpe = pnl.mean() / (pnl.std() + 1e-8)
        if annualize:
            sharpe = sharpe * math.sqrt(252.0 / FORWARD_DAYS)
    return sharpe.item()


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loaders(
    data: dict[str, torch.Tensor],
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test ``DataLoader``s from the ``build_dataset`` output dict.

    Each batch yields ``(X_temporal, y_binary, y_return)`` where:
        X_temporal : [B, window_size, n_features]
        y_binary   : [B] — 1=long, 0=short
        y_return   : [B] — raw 5-day forward log-return for loss weighting

    Args:
        data (dict):       Output of ``build_dataset`` or ``build_multi_dataset``.
        batch_size (int):  Samples per mini-batch (default 64).

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: ``(train_loader, val_loader, test_loader)``.
    """
    def _loader(split: str, shuffle: bool) -> DataLoader:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        yr_key = f"y_return_{split}"
        yr = data[yr_key] if yr_key in data else torch.zeros(X.size(0))
        ds = TensorDataset(X, y, yr)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)

    return _loader("train", True), _loader("val", False), _loader("test", False)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop when validation Sharpe stops improving.

    Args:
        patience  (int):   Epochs to wait after last improvement (default 15).
        min_delta (float): Minimum improvement to count (default 1e-3).
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best = -float("inf")

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    def __call__(self, val_sharpe: float) -> None:
        if val_sharpe > self._best + self.min_delta:
            self._best = val_sharpe
            self._counter = 0
        else:
            self._counter += 1
        log.debug("EarlyStopping counter=%d/%d", self._counter, self.patience)


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean PnL-weighted BCE loss.

    Args:
        model     (nn.Module):   Model to train.
        loader    (DataLoader):  Yields ``(X, y_binary, y_return)`` batches.
        optimizer (Optimizer):   Gradient-descent optimiser.
        device    (device):      Compute device.

    Returns:
        float: Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for xb, yb, yr_b in loader:
        xb, yb, yr_b = xb.to(device), yb.to(device), yr_b.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = pnl_weighted_bce(logits, yb, yr_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model on *loader*.

    Args:
        model  (nn.Module):  Model in eval mode.
        loader (DataLoader): Yields ``(X, y_binary, y_return)`` batches.
        device (device):     Compute device.

    Returns:
        tuple: ``(avg_loss, accuracy, sharpe)`` — accuracy is directional accuracy
               (fraction of samples where sign(signal) == label), Sharpe is annualised.
    """
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb, yr_b in loader:
            xb, yb, yr_b = xb.to(device), yb.to(device), yr_b.to(device)
            logits = model(xb)
            loss = pnl_weighted_bce(logits, yb, yr_b)
            total_loss += loss.item() * len(yb)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
            all_returns.append(yr_b.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    returns_cat = torch.cat(all_returns)

    avg_loss = total_loss / len(loader.dataset)
    preds = (logits_cat > 0).long()
    acc = (preds == labels_cat).float().mean().item()
    sharpe = compute_sharpe(logits_cat, returns_cat)
    return avg_loss, acc, sharpe


# ── Main training function ────────────────────────────────────────────────────

def train(
    model: SignalNet | None = None,
    data: dict[str, torch.Tensor] | None = None,
    ticker: str | list[str] = "AAPL",
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """End-to-end training loop: PnL-weighted BCE with early stopping on Sharpe.

    If *model* or *data* are ``None`` they are built/downloaded automatically.

    Args:
        model         (SignalNet, optional): Pre-constructed model. Defaults to build_model().
        data          (dict, optional):      Output of build_dataset(). Downloaded if None.
        ticker        (str | list[str]):     Used only when *data* is None.
        n_epochs      (int):                 Maximum training epochs (default 100).
        lr            (float):               Initial learning rate (default 3e-4).
        weight_decay  (float):               L2 regularisation (default 1e-4).
        batch_size    (int):                 Mini-batch size (default 64).
        patience      (int):                 Early-stopping patience in epochs (default 20).
        seed          (int):                 Random seed (default 42).
        checkpoint_dir (Path, optional):     Where to save ``.pt`` files. Defaults to models/.
        device        (device, optional):    Defaults to get_device().

    Returns:
        dict: Keys ``train_loss``, ``val_loss``, ``val_acc``, ``val_sharpe`` (lists per epoch).
    """
    set_seed(seed)
    if device is None:
        device = get_device()
    if checkpoint_dir is None:
        checkpoint_dir = MODELS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if data is None:
        from src.data_loader import build_dataset, build_multi_dataset
        tickers = ticker if isinstance(ticker, list) else [ticker]
        if len(tickers) == 1:
            log.info("Downloading data for %s …", tickers[0])
            data = build_dataset(tickers[0])
        else:
            log.info("Downloading data for %d tickers: %s …", len(tickers), tickers)
            data = build_multi_dataset(tickers)

    train_loader, val_loader, _ = make_loaders(data, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model is None:
        model = build_model()
    model = model.to(device)
    log.info("SignalNet — %d trainable parameters", model.count_parameters())

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    # ── Checkpoint path ───────────────────────────────────────────────────────
    if isinstance(ticker, list):
        ticker_hash = hashlib.md5(",".join(sorted(ticker)).encode()).hexdigest()[:8]
        ckpt_name = f"multi_{len(ticker)}tickers_{ticker_hash}_best.pt"
    else:
        ckpt_name = f"{ticker}_best.pt"
    best_checkpoint = checkpoint_dir / ckpt_name

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_sharpe": []
    }
    best_sharpe = -float("inf")

    log.info("Training — max %d epochs, device=%s", n_epochs, device)

    for epoch in range(1, n_epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_acc, v_sharpe = evaluate(model, val_loader, device)

        scheduler.step()
        early_stop(v_sharpe)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_sharpe"].append(v_sharpe)

        if v_sharpe > best_sharpe:
            best_sharpe = v_sharpe
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "val_sharpe": v_sharpe,
                },
                best_checkpoint,
            )
            log.info(
                "Epoch %3d | loss=%.4f  acc=%.3f  sharpe=%+.3f  ✓ saved",
                epoch, v_loss, v_acc, v_sharpe,
            )
        else:
            log.info(
                "Epoch %3d | loss=%.4f  acc=%.3f  sharpe=%+.3f",
                epoch, v_loss, v_acc, v_sharpe,
            )

        if early_stop.should_stop:
            log.info("Early stopping at epoch %d (best sharpe=%.3f)", epoch, best_sharpe)
            break

    log.info("Done. Best val_sharpe=%.3f  → %s", best_sharpe, best_checkpoint)
    return history


# ── Evaluation on held-out test set ──────────────────────────────────────────

def evaluate_test(
    model: SignalNet,
    data: dict[str, torch.Tensor],
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run final evaluation on the held-out test split.

    Args:
        model      (SignalNet): Trained model.
        data       (dict):      Must contain ``X_test``, ``y_test``, ``y_return_test``.
        batch_size (int):       Samples per mini-batch (default 128).
        device     (device):    Defaults to ``get_device()``.

    Returns:
        dict: Keys ``test_loss``, ``test_acc``, ``test_sharpe``.
    """
    if device is None:
        device = get_device()
    _, _, test_loader = make_loaders(data, batch_size=batch_size)
    model = model.to(device)
    t_loss, t_acc, t_sharpe = evaluate(model, test_loader, device)
    log.info("Test | loss=%.4f  acc=%.3f  sharpe=%+.3f", t_loss, t_acc, t_sharpe)
    return {"test_loss": t_loss, "test_acc": t_acc, "test_sharpe": t_sharpe}


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train SignalNet (PnL-optimised)")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated list, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--tickers-file", default=None, metavar="PATH",
                        help="One ticker per line. Overrides --ticker and --tickers.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Evaluate mode: load a checkpoint and report test-set metrics without training.
    parser.add_argument("--evaluate", default=None, metavar="CHECKPOINT",
                        help="Path to a .pt checkpoint. Skips training, runs test-set "
                             "evaluation on the ticker(s) specified and prints results.")
    args = parser.parse_args()

    # ── Resolve tickers ───────────────────────────────────────────────────────
    if args.tickers_file:
        tickers_path = Path(args.tickers_file)
        if not tickers_path.exists():
            raise SystemExit(f"Tickers file not found: {tickers_path}")
        ticker_arg: str | list[str] = [
            line.strip().upper()
            for line in tickers_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        log.info("Loaded %d tickers from %s", len(ticker_arg), tickers_path)
    elif args.tickers:
        ticker_arg = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        ticker_arg = args.ticker.upper()

    # ── Evaluate mode ─────────────────────────────────────────────────────────
    if args.evaluate:
        ckpt_path = Path(args.evaluate)
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

        from src.data_loader import build_dataset, build_multi_dataset
        tickers = ticker_arg if isinstance(ticker_arg, list) else [ticker_arg]
        log.info("Building test data for %d ticker(s) …", len(tickers))
        data = build_dataset(tickers[0]) if len(tickers) == 1 else build_multi_dataset(tickers)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model()
        model.load_state_dict(ckpt["model_state_dict"])

        result = evaluate_test(model, data)
        print(f"\n{'─'*40}")
        print(f"  Checkpoint : {ckpt_path.name}")
        print(f"  Tickers    : {tickers if len(tickers) <= 5 else f'{len(tickers)} tickers'}")
        print(f"{'─'*40}")
        print(f"  test_loss  : {result['test_loss']:.4f}")
        print(f"  test_acc   : {result['test_acc']:.3f}  ({result['test_acc']*100:.1f}%)")
        print(f"  test_sharpe: {result['test_sharpe']:+.3f}")
        print(f"{'─'*40}")
        raise SystemExit(0)

    # ── Train mode ────────────────────────────────────────────────────────────
    hist = train(
        ticker=ticker_arg,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(f"\nDone. Final val_sharpe={hist['val_sharpe'][-1]:.3f}")
