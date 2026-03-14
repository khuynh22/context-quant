"""
train.py — Training & validation scripts for ContextQuantFusionNet.

Responsibilities:
- Configure device (CPU / CUDA)
- Build DataLoaders for train / val / test
- Run training loop with Early Stopping and LR scheduling
- Log metrics and save model checkpoints to models/

Typical usage
-------------
>>> from src.data_loader import build_dataset
>>> from src.model import build_model
>>> from src.train import train
>>> data = build_dataset("AAPL")
>>> model = build_model()
>>> history = train(model, data, n_epochs=50)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.model import ContextQuantFusionNet, build_model
from src.utils import (
    accuracy, classification_report, compute_class_weights_tensor, get_device, set_seed
)

log = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).parent.parent / "models"


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loaders(
    data: dict[str, torch.Tensor],
    batch_size: int = 64,
    sentiment_dim: int = 3,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test ``DataLoader``s from the ``build_dataset`` output dict.

    A zero-filled sentinel sentiment tensor (neutral prior) is appended when
    the dict does not contain ``X_sentiment_*`` keys.

    Args:
        data (dict): Keys ``X_train``, ``y_train``, ``X_val``, ``y_val``, ``X_test``, ``y_test``.
        batch_size (int): Samples per mini-batch (default 64).
        sentiment_dim (int): Sentiment vector width used when no real sentiment tensors are
            present (default 3).

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: ``(train_loader, val_loader, test_loader)``.
    """
    def _loader(split: str, shuffle: bool) -> DataLoader:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        key = f"X_sentiment_{split}"
        if key in data:
            s = data[key]
        else:
            s = torch.full((X.size(0), sentiment_dim), fill_value=1.0 / sentiment_dim)
        ds = TensorDataset(X, s, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)

    train_loader = _loader("train", shuffle=True)
    val_loader = _loader("val", shuffle=False)
    test_loader = _loader("test", shuffle=False)
    return train_loader, val_loader, test_loader


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after last improvement (default 10).
        min_delta (float): Minimum decrease in loss to count as an improvement (default 1e-4).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best: float = float("inf")

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    def __call__(self, val_loss: float) -> None:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        log.debug("EarlyStopping counter=%d/%d", self._counter, self.patience)


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average cross-entropy loss.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Yields ``(X_temporal, X_sentiment, y)`` batches.
        optimizer (torch.optim.Optimizer): Gradient-descent optimizer.
        criterion (nn.Module): Loss function, typically ``nn.CrossEntropyLoss``.
        device (torch.device): Device to run on.

    Returns:
        float: Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for xb, sb, yb in loader:
        xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb, sb)                # [B, 5]
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate *model* on *loader*.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Yields ``(X_temporal, X_sentiment, y)`` batches.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple[float, float]: ``(avg_loss, accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            logits = model(xb, sb)
            total_loss += criterion(logits, yb).item() * len(yb)
            all_preds.append(logits.cpu())
            all_labels.append(yb.cpu())

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy(preds_cat, labels_cat)
    return avg_loss, acc


# ── Main training function ────────────────────────────────────────────────────

def train(
    model: ContextQuantFusionNet | None = None,
    data: dict[str, torch.Tensor] | None = None,
    ticker: str | list[str] = "AAPL",
    n_epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    min_delta: float = 1e-4,
    use_class_weights: bool = True,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """End-to-end training loop with early stopping and LR scheduling.

    If *model* or *data* are ``None`` they are built/downloaded automatically.

    Args:
        model (ContextQuantFusionNet, optional): Pre-constructed model. Defaults to build_model().
        data (dict, optional): Output of build_dataset(ticker). Downloaded if None.
        ticker (str): Used only when *data* is None.
        n_epochs (int): Maximum training epochs (default 60).
        lr (float): Initial learning rate for Adam (default 1e-3).
        batch_size (int): Mini-batch size (default 64).
        patience (int): Early stopping patience in epochs (default 10).
        min_delta (float): Minimum improvement threshold for early stopping (default 1e-4).
        use_class_weights (bool): Pass balanced class weights to CrossEntropyLoss (default True).
        seed (int): Random seed (default 42).
        checkpoint_dir (Path, optional): Directory to save .pt checkpoints.
            Defaults to <repo>/models/.
        device (torch.device, optional): Defaults to get_device().

    Returns:
        dict: Keys ``train_loss``, ``val_loss``, ``val_acc`` (lists per epoch).
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
    log.info("Model: %d trainable parameters", model.count_parameters())

    # ── Loss ──────────────────────────────────────────────────────────────────
    if use_class_weights:
        w = compute_class_weights_tensor(data["y_train"].numpy()).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

    # ── History ───────────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    if isinstance(ticker, list):
        ticker_hash = hashlib.md5(",".join(sorted(ticker)).encode()).hexdigest()[:8]
        ckpt_name = f"multi_{len(ticker)}tickers_{ticker_hash}_best.pt"
    else:
        ckpt_name = f"{ticker}_best.pt"
    best_checkpoint = checkpoint_dir / ckpt_name

    log.info("Starting training — max %d epochs, device=%s", n_epochs, device)

    for epoch in range(1, n_epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(v_loss)
        early_stop(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                },
                best_checkpoint,
            )
            log.info(
                "Epoch %3d | train_loss=%.4f val_loss=%.4f val_acc=%.3f  ✓ saved",
                epoch, t_loss, v_loss, v_acc
            )
        else:
            log.info(
                "Epoch %3d | train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                epoch, t_loss, v_loss, v_acc
            )

        if early_stop.should_stop:
            log.info("Early stopping triggered at epoch %d", epoch)
            break

    log.info(
        "Training complete. Best val_loss=%.4f  checkpoint → %s", best_val_loss, best_checkpoint
    )
    return history


# ── Evaluation on held-out test set ──────────────────────────────────────────

def evaluate_test(
    model: ContextQuantFusionNet,
    data: dict[str, torch.Tensor],
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run final evaluation on the test split.

    Args:
        model (ContextQuantFusionNet): The trained model.
        data (dict): Must contain ``X_test``, ``y_test`` (and optionally ``X_sentiment_test``).
        batch_size (int): Samples per mini-batch (default 128).
        device (torch.device, optional): Defaults to ``get_device()``.

    Returns:
        dict: Keys ``test_loss``, ``test_acc``, and ``report`` (str).
    """
    if device is None:
        device = get_device()
    _, _, test_loader = make_loaders(data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    t_loss, t_acc = evaluate(model, test_loader, criterion, device)

    # Gather all preds for classification_report
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, sb, yb in test_loader:
            logits = model(xb.to(device), sb.to(device))
            all_preds.append(logits.cpu())
            all_labels.append(yb.cpu())

    report = classification_report(torch.cat(all_preds), torch.cat(all_labels))
    log.info("Test  | loss=%.4f  acc=%.3f", t_loss, t_acc)
    return {"test_loss": t_loss, "test_acc": t_acc, "report": report}


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train ContextQuantFusionNet")
    parser.add_argument("--ticker", default="AAPL",
                        help="Single ticker symbol (default: AAPL)")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated list of tickers, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--tickers-file", default=None, metavar="PATH",
                        help="Path to a text file with one ticker per line. "
                             "Blank lines and lines starting with # are ignored. "
                             "Overrides --ticker and --tickers.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

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

    hist = train(
        ticker=ticker_arg,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    final_acc = hist["val_acc"][-1]
    print(f"\nFinished. Final val_acc={final_acc:.3f}")
