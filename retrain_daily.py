"""
retrain_daily.py — Scheduled daily retraining script for ContextQuantFusionNet.

Run manually:
    uv run python retrain_daily.py

Or let Windows Task Scheduler run it every morning at 6 AM (before market open).
See schedule_task.ps1 to register the scheduled task automatically.

What this does each run:
1. Reads tickers from tickers.txt (or --tickers-file argument)
2. Downloads fresh data up to today for every ticker
3. Retrains the model on the full updated dataset
4. Saves a new checkpoint to models/ (overwrites previous best)
5. Writes a log entry to logs/retrain.log
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Logging setup — writes to both console and a rolling log file ─────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "retrain.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
DEFAULT_TICKERS_FILE = REPO_ROOT / "tickers.txt"


def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    tickers = [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily ContextQuant retraining job")
    parser.add_argument(
        "--tickers-file", default=str(DEFAULT_TICKERS_FILE), metavar="PATH",
        help="Text file with one ticker per line (default: tickers.txt)",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_start = datetime.now()
    log.info("=" * 60)
    log.info("ContextQuant daily retrain — %s", run_start.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # ── Load tickers ──────────────────────────────────────────────────────────
    tickers_path = Path(args.tickers_file)
    tickers = load_tickers(tickers_path)
    log.info("Universe: %d tickers from %s", len(tickers), tickers_path.name)
    log.info("Tickers: %s", ", ".join(tickers))

    # ── Train ─────────────────────────────────────────────────────────────────
    from src.train import train  # import here so logging is already set up

    try:
        history = train(
            ticker=tickers,          # list triggers build_multi_dataset
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    except Exception as exc:
        log.exception("Training failed: %s", exc)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("nan")
    elapsed = (datetime.now() - run_start).total_seconds()

    log.info("-" * 60)
    log.info("Done in %.0f s | best_val_loss=%.4f | final_val_acc=%.3f",
             elapsed, best_val_loss, final_val_acc)
    log.info("Checkpoint saved to models/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
