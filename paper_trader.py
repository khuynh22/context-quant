"""
paper_trader.py — Paper trading simulation engine for ContextQuant.

Simulates a brokerage account with fake money using real market prices
from Yahoo Finance. No real orders are ever placed.

How it works
------------
Each day you call `run_daily()`:
  1. The SignalEngine scans all tickers, runs dual-head inference, and applies
     confidence + return-sign filtering.
  2. Actionable BUY/SELL signals open new positions sized by the model's ATR-based
     position sizing (higher confidence + lower volatility → larger size).
  3. Existing positions are checked for exit conditions using the stop-loss and
     take-profit levels set at entry (derived from ATR at open time).
  4. All activity is logged to paper_trading/journal.csv.
  5. Portfolio state is saved to paper_trading/portfolio.json.

Usage
-----
    uv run python paper_trader.py              # run today's session
    uv run python paper_trader.py --report     # print performance summary
    uv run python paper_trader.py --reset      # wipe portfolio and start fresh
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    FEATURE_COLS,
    SECTOR_MAP,
    WINDOW_SIZE,
    add_technical_indicators,
    download_ohlcv,
    _get_spy_context,
)
from src.signal_engine import SignalEngine, TradeSignal

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "models"
PT_DIR = REPO_ROOT / "paper_trading"
PORTFOLIO_FILE = PT_DIR / "portfolio.json"
JOURNAL_FILE = PT_DIR / "journal.csv"
TICKERS_FILE = REPO_ROOT / "tickers.txt"

# ── Sizing / risk constants ────────────────────────────────────────────────────
STARTING_CASH = 10_000.00    # fake dollars
MAX_POSITION_PCT = 0.10      # SignalEngine caps position at this fraction of portfolio
MAX_HOLD_DAYS = 5            # force-exit after N trading days regardless of P&L


# ── Portfolio state ───────────────────────────────────────────────────────────

def _empty_portfolio() -> dict:
    return {
        "cash":         STARTING_CASH,
        "positions":    {},   # ticker → position dict
        "closed_trades": [],
        "created":      str(date.today()),
    }


def load_portfolio() -> dict:
    PT_DIR.mkdir(exist_ok=True)
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return _empty_portfolio()


def save_portfolio(portfolio: dict) -> None:
    PT_DIR.mkdir(exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))


# ── Journal (append-only CSV) ─────────────────────────────────────────────────

_JOURNAL_COLS = [
    "date", "action", "ticker", "side", "confidence", "predicted_return",
    "price", "shares", "notional", "pnl", "pnl_pct", "reason", "cash_after",
]


def _append_journal(row: dict) -> None:
    PT_DIR.mkdir(exist_ok=True)
    write_header = not JOURNAL_FILE.exists()
    with open(JOURNAL_FILE, "a", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=_JOURNAL_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Market data helpers ───────────────────────────────────────────────────────

def get_current_price(ticker: str) -> float | None:
    """Fetch today's closing price (or most recent available)."""
    try:
        df = download_ohlcv(ticker)
        return float(df["Close"].iloc[-1])
    except Exception as exc:
        log.warning("Could not fetch price for %s: %s", ticker, exc)
        return None


def build_inference_inputs(
    ticker: str,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Build model inputs for a single ticker.

    Returns:
        ``(x_temporal [1, 60, 14], x_sector [1], atr_pct)`` or ``None`` on failure.
        The scaler is fit on all available data (no train/val split for live inference).
    """
    try:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = "2010-01-01"
        df = download_ohlcv(ticker, start=start, end=end)
        df = add_technical_indicators(df)

        # Join SPY market regime features
        spy_ctx = _get_spy_context(df.index, start=start, end=end)
        df = df.join(spy_ctx, how="left")
        df = df.dropna(subset=FEATURE_COLS)

        if len(df) < WINDOW_SIZE:
            log.warning("Too few rows for %s after indicators (%d)", ticker, len(df))
            return None

        # Grab ATR% from the most recent row (before scaling)
        atr_pct = float(df["atr_pct"].iloc[-1])

        # Fit StandardScaler on all available history, scale the last window
        X_all = df[FEATURE_COLS].values.astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        window = X_scaled[-WINDOW_SIZE:]                         # [60, 14]

        x_temporal = torch.tensor(window[np.newaxis], dtype=torch.float32)  # [1, 60, 14]
        sector_id = SECTOR_MAP.get(ticker.upper(), 8)
        x_sector = torch.tensor([sector_id], dtype=torch.long)  # [1]

        return x_temporal, x_sector, atr_pct
    except Exception as exc:
        log.warning("build_inference_inputs failed for %s: %s", ticker, exc)
        return None


# ── Model helpers ─────────────────────────────────────────────────────────────

def find_latest_checkpoint() -> Path | None:
    pts = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


# ── Core trading logic ────────────────────────────────────────────────────────

def _portfolio_value(portfolio: dict) -> float:
    """Cash + entry-price value of open positions (conservative proxy)."""
    open_value = sum(
        p["shares"] * p["entry_price"] for p in portfolio["positions"].values()
    )
    return portfolio["cash"] + open_value


def _open_position(
    portfolio: dict,
    ticker: str,
    signal: TradeSignal,
    price: float,
    today: str,
) -> None:
    """Open a new position sized by the signal's position_pct."""
    portfolio_value = _portfolio_value(portfolio)
    # signal.position_pct is already capped at max_position by SignalEngine
    notional = round(portfolio_value * signal.position_pct, 2)
    notional = min(notional, portfolio["cash"] * 0.95)   # never exceed available cash
    if notional < 1.0:
        log.info("  Not enough cash to open %s", ticker)
        return

    shares = notional / price
    portfolio["cash"] -= notional
    portfolio["positions"][ticker] = {
        "side":             "long" if signal.action == "BUY" else "short",
        "confidence":       signal.confidence,
        "predicted_return": signal.predicted_return,
        "entry_price":      price,
        "shares":           shares,
        "notional":         notional,
        "entry_date":       today,
        "hold_days":        0,
        "stop_loss_pct":    signal.stop_loss_pct,
        "take_profit_pct":  signal.take_profit_pct,
    }
    _append_journal({
        "date": today, "action": "OPEN", "ticker": ticker,
        "side": portfolio["positions"][ticker]["side"],
        "confidence": round(signal.confidence, 4),
        "predicted_return": round(signal.predicted_return, 4),
        "price": round(price, 4), "shares": round(shares, 6),
        "notional": round(notional, 2), "pnl": 0, "pnl_pct": 0,
        "reason": "signal", "cash_after": round(portfolio["cash"], 2),
    })
    log.info(
        "  OPEN  %-6s %-5s  $%.2f @ $%.2f  conf=%.0f%%  ret_pred=%+.1f%%  "
        "SL=%.1f%%  TP=%.1f%%",
        ticker, portfolio["positions"][ticker]["side"].upper(),
        notional, price, signal.confidence * 100, signal.predicted_return * 100,
        signal.stop_loss_pct * 100, signal.take_profit_pct * 100,
    )


def _close_position(
    portfolio: dict,
    ticker: str,
    current_price: float,
    reason: str,
    today: str,
) -> None:
    """Close an existing position and realise P&L."""
    pos = portfolio["positions"].pop(ticker)
    shares = pos["shares"]
    entry = pos["entry_price"]

    if pos["side"] == "long":
        proceeds = shares * current_price
        pnl = proceeds - pos["notional"]
    else:  # short
        proceeds = pos["notional"] + (entry - current_price) * shares
        pnl = proceeds - pos["notional"]

    portfolio["cash"] += proceeds
    pnl_pct = pnl / pos["notional"]

    portfolio["closed_trades"].append({
        "ticker":           ticker,
        "side":             pos["side"],
        "entry_date":       pos["entry_date"],
        "exit_date":        today,
        "entry_price":      entry,
        "exit_price":       current_price,
        "pnl":              round(pnl, 2),
        "pnl_pct":          round(pnl_pct, 4),
    })
    _append_journal({
        "date": today, "action": "CLOSE", "ticker": ticker,
        "side": pos["side"],
        "confidence": pos["confidence"],
        "predicted_return": pos.get("predicted_return", 0),
        "price": round(current_price, 4), "shares": round(shares, 6),
        "notional": round(proceeds, 2), "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 4), "reason": reason,
        "cash_after": round(portfolio["cash"], 2),
    })
    emoji = "✅" if pnl >= 0 else "❌"
    log.info(
        "  CLOSE %-6s %-5s  P&L $%+.2f (%+.1f%%)  [%s]  %s",
        ticker, pos["side"].upper(), pnl, pnl_pct * 100, reason, emoji,
    )


# ── Main daily session ────────────────────────────────────────────────────────

def run_daily(tickers: list[str], engine: SignalEngine) -> None:
    today = str(date.today())
    portfolio = load_portfolio()

    print(f"\n{'═'*64}")
    print(f"  ContextQuant Paper Trading — {today}")
    print(f"  Cash: ${portfolio['cash']:>10,.2f}   "
          f"Portfolio: ${_portfolio_value(portfolio):>10,.2f}")
    print(f"  Open positions: {len(portfolio['positions'])}")
    print(f"{'═'*64}")

    # ── Step 1: Check exits on existing positions ─────────────────────────────
    if portfolio["positions"]:
        print("\n── Checking open positions ──────────────────────────────────────")
    for ticker in list(portfolio["positions"].keys()):
        pos = portfolio["positions"][ticker]
        price = get_current_price(ticker)
        if price is None:
            continue

        pos["hold_days"] = pos.get("hold_days", 0) + 1
        if pos["side"] == "long":
            ret = (price - pos["entry_price"]) / pos["entry_price"]
        else:
            ret = (pos["entry_price"] - price) / pos["entry_price"]

        sl = pos.get("stop_loss_pct", 0.03)
        tp = pos.get("take_profit_pct", 0.06)

        if ret <= -sl:
            _close_position(portfolio, ticker, price, "stop_loss", today)
        elif ret >= tp:
            _close_position(portfolio, ticker, price, "take_profit", today)
        elif pos["hold_days"] >= MAX_HOLD_DAYS:
            _close_position(portfolio, ticker, price, "max_hold", today)
        else:
            log.info(
                "  HOLD  %-6s  return %+.1f%%  day %d/%d  SL=%.1f%%  TP=%.1f%%",
                ticker, ret * 100, pos["hold_days"], MAX_HOLD_DAYS, sl * 100, tp * 100,
            )

    # ── Step 2: Scan for new signals ──────────────────────────────────────────
    print(f"\n── Scanning {len(tickers)} tickers for new signals ──────────────────")
    new_signals: list[tuple[str, TradeSignal, float]] = []

    for ticker in tickers:
        if ticker in portfolio["positions"]:
            continue   # already holding, skip
        inputs = build_inference_inputs(ticker)
        if inputs is None:
            continue
        x_temporal, x_sector, atr_pct = inputs
        signal = engine.generate(x_temporal, x_sector, current_atr_pct=atr_pct)
        if signal.action != "HOLD":
            new_signals.append((ticker, signal, atr_pct))
            log.debug("  %s → %s", ticker, signal)

    # Sort by confidence descending
    new_signals.sort(key=lambda r: -r[1].confidence)

    if not new_signals:
        print("  No signals meet the confidence threshold today.")
    else:
        print(f"  {len(new_signals)} actionable signal(s) found:")
        for ticker, signal, _ in new_signals:
            price = get_current_price(ticker)
            if price is None:
                continue
            _open_position(portfolio, ticker, signal, price, today)

    # ── Step 3: Save and print summary ───────────────────────────────────────
    save_portfolio(portfolio)
    _print_summary(portfolio)


def _print_summary(portfolio: dict) -> None:
    total_value = _portfolio_value(portfolio)
    total_pnl = total_value - STARTING_CASH
    total_pnl_pct = total_pnl / STARTING_CASH * 100
    closed = portfolio["closed_trades"]
    wins = sum(1 for t in closed if t["pnl"] > 0)
    losses = sum(1 for t in closed if t["pnl"] <= 0)
    win_rate = (wins / len(closed) * 100) if closed else 0

    print("\n── Portfolio Summary ─────────────────────────────────────────────")
    print(f"  Total value:    ${total_value:>10,.2f}  (started ${STARTING_CASH:,.2f})")
    print(f"  Total P&L:      ${total_pnl:>+10,.2f}  ({total_pnl_pct:+.2f}%)")
    print(f"  Open positions: {len(portfolio['positions'])}")
    print(f"  Closed trades:  {len(closed)}  ({wins} wins / {losses} losses)")
    if closed:
        print(f"  Win rate:       {win_rate:.1f}%")
    if portfolio["positions"]:
        print(f"\n  {'TICKER':<8} {'SIDE':<6} {'ENTRY':>8}  {'HELD':>5}  "
              f"{'SL':>6}  {'TP':>6}")
        for ticker, pos in portfolio["positions"].items():
            print(
                f"  {ticker:<8} {pos['side']:<6} "
                f"${pos['entry_price']:>7.2f}  {pos['hold_days']:>4}d  "
                f"{pos.get('stop_loss_pct', 0)*100:>5.1f}%  "
                f"{pos.get('take_profit_pct', 0)*100:>5.1f}%"
            )
    print(f"{'═'*64}\n")


# ── Performance report ────────────────────────────────────────────────────────

def print_report() -> None:
    if not JOURNAL_FILE.exists():
        print("No journal data yet. Run a session first.")
        return

    df = pd.read_csv(JOURNAL_FILE)
    closed = df[df["action"] == "CLOSE"].copy()

    if closed.empty:
        print("No closed trades yet.")
        return

    total_pnl = closed["pnl"].sum()
    win_rate = (closed["pnl"] > 0).mean() * 100
    avg_win = closed.loc[closed["pnl"] > 0, "pnl"].mean()
    avg_loss = closed.loc[closed["pnl"] <= 0, "pnl"].mean()
    best_trade = closed.loc[closed["pnl"].idxmax()]
    worst_trade = closed.loc[closed["pnl"].idxmin()]

    print(f"\n{'═'*64}")
    print("  ContextQuant Paper Trading Performance Report")
    print(f"  Period: {df['date'].min()} → {df['date'].max()}")
    print(f"{'─'*64}")
    print(f"  Total closed trades : {len(closed)}")
    print(f"  Total P&L           : ${total_pnl:>+,.2f}")
    print(f"  Win rate            : {win_rate:.1f}%")
    print(f"  Avg winning trade   : ${avg_win:>+,.2f}")
    print(f"  Avg losing trade    : ${avg_loss:>+,.2f}")
    print(f"  Best trade          : {best_trade['ticker']}  ${best_trade['pnl']:+,.2f}")
    print(f"  Worst trade         : {worst_trade['ticker']}  ${worst_trade['pnl']:+,.2f}")
    print(f"{'─'*64}")

    by_reason = closed.groupby("reason")["pnl"].agg(["count", "sum", "mean"])
    print("\n  Exits by reason:")
    for reason, row in by_reason.iterrows():
        print(f"    {reason:<12}  n={int(row['count'])}  "
              f"total=${row['sum']:>+,.2f}  avg=${row['mean']:>+,.2f}")

    by_ticker = closed.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    print("\n  P&L by ticker (top 10):")
    max_abs = max(abs(by_ticker).max(), 1)
    for ticker, pnl in by_ticker.head(10).items():
        bar = "█" * int(abs(pnl) / max_abs * 20)
        sign = "+" if pnl >= 0 else "-"
        print(f"    {ticker:<8}  {sign}${abs(pnl):>7.2f}  {bar}")
    print(f"{'═'*64}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tickers(path: Path) -> list[str]:
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ContextQuant paper trading engine")
    parser.add_argument("--tickers-file", default=str(TICKERS_FILE))
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt file. Defaults to most recent in models/")
    parser.add_argument("--conf-threshold", type=float, default=0.60,
                        help="Minimum directional probability to act (default: 0.60)")
    parser.add_argument("--max-position", type=float, default=MAX_POSITION_PCT,
                        help="Max fraction of portfolio per trade (default: 0.10)")
    parser.add_argument("--report", action="store_true",
                        help="Print performance report and exit")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe portfolio and start fresh with $10,000")
    args = parser.parse_args()

    if args.reset:
        confirm = input("Reset portfolio and lose all history? [y/N] ").strip().lower()
        if confirm == "y":
            if PORTFOLIO_FILE.exists():
                PORTFOLIO_FILE.unlink()
            if JOURNAL_FILE.exists():
                JOURNAL_FILE.unlink()
            print(f"Portfolio reset. Starting cash: ${STARTING_CASH:,.2f}")
        else:
            print("Aborted.")
        raise SystemExit(0)

    if args.report:
        print_report()
        raise SystemExit(0)

    # ── Normal daily run ──────────────────────────────────────────────────────
    tickers = load_tickers(Path(args.tickers_file))
    log.info("Loaded %d tickers", len(tickers))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(
            "No checkpoint found. Train the model first:\n"
            "  uv run python -m src.train --tickers-file tickers.txt"
        )
    log.info("Checkpoint: %s", ckpt_path.name)

    engine = SignalEngine.from_checkpoint(
        ckpt_path,
        conf_threshold=args.conf_threshold,
        max_position=args.max_position,
    )
    log.info(
        "SignalEngine ready  conf_threshold=%.2f  max_position=%.0f%%",
        args.conf_threshold, args.max_position * 100,
    )

    run_daily(tickers, engine)
