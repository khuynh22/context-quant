"""
app.py — ContextQuant Streamlit Dashboard.

Run with:
    uv run streamlit run app.py

Tabs
----
Signal      Pick any ticker → live BUY / SELL / HOLD signal with price chart
Training    Train a model in-browser, watch loss / accuracy / Sharpe per epoch
Portfolio   Paper trading journal, equity curve, win-rate breakdown
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    FEATURE_COLS,
    WINDOW_SIZE,
    add_technical_indicators,
    download_ohlcv,
    _get_spy_context,
)
from src.signal_engine import SignalEngine, TradeSignal

MODELS_DIR = Path(__file__).parent / "models"
PT_DIR = Path(__file__).parent / "paper_trading"
JOURNAL_FILE = PT_DIR / "journal.csv"
PORTFOLIO_FILE = PT_DIR / "portfolio.json"

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextQuant",
    page_icon="📈",
    layout="wide",
)

ACTION_COLOR = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#888888"}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_engine(checkpoint_name: str, edge_threshold: float) -> SignalEngine:
    path = MODELS_DIR / checkpoint_name
    return SignalEngine.from_checkpoint(path, edge_threshold=edge_threshold)


@st.cache_data(show_spinner="Fetching market data …")
def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = download_ohlcv(ticker, start="2010-01-01", end=end)
        df = add_technical_indicators(df)
        spy = _get_spy_context(df.index, start="2010-01-01", end=end)
        df = df.join(spy, how="left").dropna(subset=FEATURE_COLS)
        return df
    except Exception as exc:
        st.error(f"Could not fetch {ticker}: {exc}")
        return None


def build_window(df: pd.DataFrame) -> torch.Tensor | None:
    """Scale all available history, return the last WINDOW_SIZE rows as a tensor."""
    if len(df) < WINDOW_SIZE:
        st.warning(f"Need at least {WINDOW_SIZE} rows; got {len(df)}.")
        return None
    X = df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    window = X_scaled[-WINDOW_SIZE:]
    return torch.tensor(window[np.newaxis], dtype=torch.float32)  # [1, 20, 14]


def find_checkpoints() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.glob("*.pt")], reverse=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_signal, tab_train, tab_portfolio = st.tabs(["📡 Signal", "🏋️ Training", "💼 Portfolio"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_signal:
    st.header("Live Signal")
    st.caption(
        "Loads a trained checkpoint, runs the last 20 days of market data through "
        "SignalNet, and shows the BUY / SELL / HOLD decision."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        ticker = st.text_input("Ticker", value="AAPL", key="sig_ticker").upper().strip()
        checkpoints = find_checkpoints()
        if not checkpoints:
            st.warning("No trained models found. Go to the **Training** tab first.")
            selected_ckpt = None
        else:
            selected_ckpt = st.selectbox("Checkpoint", checkpoints, key="sig_ckpt")

        edge_threshold = st.slider(
            "Edge threshold",
            min_value=0.05, max_value=0.45, value=0.20, step=0.05,
            help="Act only when |signed_signal| exceeds this. "
                 "0.20 means P(long) > 0.60 or < 0.40.",
            key="sig_thresh",
        )
        run_btn = st.button("Generate signal", type="primary", use_container_width=True)

    with col_right:
        if run_btn and selected_ckpt:
            df = fetch_data(ticker)
            if df is not None:
                x = build_window(df)
                if x is not None:
                    engine = load_engine(selected_ckpt, edge_threshold)
                    atr_pct = float(df["atr_pct"].iloc[-1])
                    signal: TradeSignal = engine.generate(x, current_atr_pct=atr_pct)

                    color = ACTION_COLOR[signal.action]

                    # Action banner
                    st.markdown(
                        f"""
                        <div style="background:{color}22;border-left:6px solid {color};
                                    padding:1rem 1.5rem;border-radius:6px;margin-bottom:1rem">
                            <span style="font-size:2.5rem;font-weight:800;color:{color}">
                                {signal.action}
                            </span>
                            <span style="margin-left:1rem;font-size:1rem;color:#888">
                                {ticker} · {df.index[-1].date()}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("P(long)", f"{signal.confidence:.1%}")
                    m2.metric("Signed signal", f"{signal.signed_signal:+.3f}")
                    m3.metric("Position size", f"{signal.position_pct:.1%}")
                    m4.metric("ATR", f"{atr_pct:.2%}")

                    m5, m6 = st.columns(2)
                    m5.metric("Stop loss", f"{signal.stop_loss_pct:.2%}")
                    m6.metric("Take profit", f"{signal.take_profit_pct:.2%}")

                    # Gauge — P(long) as a horizontal bar
                    st.markdown("**Model conviction**")
                    p = signal.confidence
                    bar_html = f"""
                    <div style="background:#eee;border-radius:4px;height:20px;
                                width:100%;position:relative">
                        <div style="width:{p*100:.1f}%;background:{color};height:100%;
                                    border-radius:4px"></div>
                        <span style="position:absolute;left:50%;top:0;
                                     transform:translateX(-50%);
                                     font-size:0.75rem;line-height:20px;color:#333">
                            P(long) = {p:.1%}
                        </span>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)

                    # Price chart — last 60 days
                    st.markdown("**Recent price (Close)**")
                    chart_data = df[["Close"]].tail(60).copy()
                    st.line_chart(chart_data, height=200)

                    # Feature snapshot
                    with st.expander("Last 5 rows of features"):
                        st.dataframe(
                            df[["Close", "RSI_14", "MACD", "bb_position", "atr_pct",
                                "spy_ret", "spy_rsi"]].tail(5).round(4),
                            use_container_width=True,
                        )
        elif run_btn and not selected_ckpt:
            st.warning("Train a model first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.header("Train SignalNet")
    st.caption(
        "PnL-weighted BCE loss. Checkpoint saved on best **val Sharpe**, not val loss. "
        "Early stopping watches Sharpe."
    )

    col_cfg, col_run = st.columns([1, 2])

    with col_cfg:
        train_ticker = st.text_input("Ticker (single)", value="AAPL", key="tr_ticker").upper()
        epochs = st.number_input("Max epochs", min_value=10, max_value=300, value=100, step=10)
        patience = st.number_input("Early stop patience", min_value=5, max_value=50, value=20)
        lr = st.select_slider(
            "Learning rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            value=3e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        train_btn = st.button("Start training", type="primary", use_container_width=True)

    with col_run:
        if train_btn:
            loss_ph = st.empty()
            chart_ph = st.empty()
            status_ph = st.empty()

            history: dict[str, list[float]] = {
                "train_loss": [], "val_loss": [], "val_acc": [], "val_sharpe": []
            }

            # Monkey-patch the training loop to stream results epoch-by-epoch
            # by running with n_epochs=1 in a loop (simple approach)
            import src.train as _train_mod
            from src.data_loader import build_dataset
            from src.model import build_model as _build_model
            from src.utils import get_device, set_seed

            set_seed(42)
            device = get_device()

            with st.spinner("Downloading data …"):
                data = build_dataset(train_ticker)

            model = _build_model().to(device)
            train_loader, val_loader, _ = _train_mod.make_loaders(data, batch_size=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-6)
            early_stop = _train_mod.EarlyStopping(patience=int(patience))

            MODELS_DIR.mkdir(exist_ok=True)
            ckpt_path = MODELS_DIR / f"{train_ticker}_best.pt"
            best_sharpe = -float("inf")

            progress = st.progress(0, text="Epoch 0")

            for epoch in range(1, int(epochs) + 1):
                t_loss = _train_mod.train_epoch(model, train_loader, optimizer, device)
                v_loss, v_acc, v_sharpe = _train_mod.evaluate(model, val_loader, device)
                scheduler.step()
                early_stop(v_sharpe)

                history["train_loss"].append(t_loss)
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                history["val_sharpe"].append(v_sharpe)

                if v_sharpe > best_sharpe:
                    best_sharpe = v_sharpe
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": v_loss, "val_acc": v_acc, "val_sharpe": v_sharpe,
                    }, ckpt_path)

                progress.progress(
                    epoch / int(epochs),
                    text=f"Epoch {epoch}/{int(epochs)} | "
                         f"loss={v_loss:.4f}  acc={v_acc:.3f}  sharpe={v_sharpe:+.3f}"
                         + (" ✓" if v_sharpe == best_sharpe else ""),
                )

                # Live charts
                hist_df = pd.DataFrame(history)
                with chart_ph.container():
                    c1, c2 = st.columns(2)
                    c1.subheader("Loss")
                    c1.line_chart(hist_df[["train_loss", "val_loss"]], height=200)
                    c2.subheader("Val Sharpe")
                    c2.line_chart(hist_df[["val_sharpe"]], height=200)

                    c3, c4 = st.columns(2)
                    c3.subheader("Val Accuracy")
                    c3.line_chart(hist_df[["val_acc"]], height=180)

                if early_stop.should_stop:
                    status_ph.success(
                        f"Early stopping at epoch {epoch}. "
                        f"Best val Sharpe = **{best_sharpe:.3f}** → saved to `{ckpt_path.name}`"
                    )
                    break
            else:
                status_ph.success(
                    f"Training complete ({int(epochs)} epochs). "
                    f"Best val Sharpe = **{best_sharpe:.3f}** → `{ckpt_path.name}`"
                )

            # Final summary table
            st.subheader("Epoch history")
            display_hist = pd.DataFrame(history)
            display_hist.index += 1
            display_hist.index.name = "epoch"
            st.dataframe(display_hist.round(4), use_container_width=True)

    # ── Test set evaluation ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Test Set Evaluation")
    st.caption(
        "The test set is the **honest number** — held out entirely during training "
        "and checkpoint selection. Run this after training to verify the model "
        "generalises and isn't just overfitting to the validation set."
    )

    eval_col1, eval_col2 = st.columns([1, 2])
    with eval_col1:
        eval_checkpoints = find_checkpoints()
        if eval_checkpoints:
            eval_ckpt = st.selectbox("Checkpoint to evaluate", eval_checkpoints, key="eval_ckpt")
            eval_ticker = st.text_input(
                "Ticker(s) for test data",
                value="AAPL",
                key="eval_ticker",
                help="Single ticker, or comma-separated list matching what was trained on.",
            )
            eval_btn = st.button("Run test evaluation", use_container_width=True)
        else:
            st.info("Train a model first.")
            eval_btn = False

    with eval_col2:
        if eval_btn and eval_checkpoints:
            from src.data_loader import build_dataset, build_multi_dataset
            from src.train import evaluate_test as _eval_test

            tickers_input = [t.strip().upper() for t in eval_ticker.split(",") if t.strip()]

            with st.spinner("Building test data …"):
                if len(tickers_input) == 1:
                    eval_data = build_dataset(tickers_input[0])
                else:
                    eval_data = build_multi_dataset(tickers_input)

            with st.spinner("Running evaluation …"):
                ckpt = torch.load(MODELS_DIR / eval_ckpt, map_location="cpu")
                from src.model import build_model as _bm
                eval_model = _bm()
                eval_model.load_state_dict(ckpt["model_state_dict"])
                result = _eval_test(eval_model, eval_data)

            val_sharpe = ckpt.get("val_sharpe", None)

            r1, r2, r3 = st.columns(3)
            r1.metric("Test loss", f"{result['test_loss']:.4f}")
            r2.metric("Test accuracy", f"{result['test_acc']:.1%}")

            # Show delta vs val Sharpe if available
            if val_sharpe is not None:
                delta = result["test_sharpe"] - val_sharpe
                r3.metric(
                    "Test Sharpe",
                    f"{result['test_sharpe']:+.3f}",
                    delta=f"{delta:+.3f} vs val",
                    delta_color="normal",
                )
            else:
                r3.metric("Test Sharpe", f"{result['test_sharpe']:+.3f}")

            # Verdict banner
            ts = result["test_sharpe"]
            if ts >= 0.7:
                st.success(
                    f"Test Sharpe {ts:+.3f} — model generalises well. "
                    "The paper trader signal is credible."
                )
            elif ts >= 0.3:
                st.warning(
                    f"Test Sharpe {ts:+.3f} — moderate. Some signal, but treat "
                    "paper trading results cautiously."
                )
            else:
                st.error(
                    f"Test Sharpe {ts:+.3f} — poor generalisation. "
                    "Val Sharpe was likely overfitted. Retrain or reconsider features."
                )

            if val_sharpe is not None:
                st.caption(
                    f"Val Sharpe at checkpoint: **{val_sharpe:+.3f}**  →  "
                    f"Test Sharpe: **{ts:+.3f}**  "
                    f"(drop: {val_sharpe - ts:+.3f})"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("Paper Trading Portfolio")

    if not JOURNAL_FILE.exists():
        st.info(
            "No trading journal found. Run the paper trader first:\n\n"
            "```\nuv run python paper_trader.py\n```"
        )
    else:
        journal = pd.read_csv(JOURNAL_FILE, parse_dates=["date"])
        closed = journal[journal["action"] == "CLOSE"].copy()
        opens = journal[journal["action"] == "OPEN"].copy()

        # ── Portfolio state ───────────────────────────────────────────────────
        if PORTFOLIO_FILE.exists():
            portfolio = json.loads(PORTFOLIO_FILE.read_text())
            starting = 10_000.0
            cash = portfolio.get("cash", starting)
            open_value = sum(
                p["shares"] * p["entry_price"]
                for p in portfolio.get("positions", {}).values()
            )
            total_value = cash + open_value
            total_pnl = total_value - starting
            total_pnl_pct = total_pnl / starting

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Portfolio value", f"${total_value:,.2f}")
            m2.metric("Total P&L", f"${total_pnl:+,.2f}", f"{total_pnl_pct:+.2%}")
            m3.metric("Cash", f"${cash:,.2f}")
            m4.metric("Open positions", len(portfolio.get("positions", {})))

        st.divider()

        if closed.empty:
            st.info("No closed trades yet.")
        else:
            # ── KPIs ──────────────────────────────────────────────────────────
            win_rate = (closed["pnl"] > 0).mean()
            avg_win = closed.loc[closed["pnl"] > 0, "pnl"].mean()
            avg_loss = closed.loc[closed["pnl"] <= 0, "pnl"].mean()
            profit_factor = (
                abs(closed.loc[closed["pnl"] > 0, "pnl"].sum())
                / (abs(closed.loc[closed["pnl"] <= 0, "pnl"].sum()) + 1e-9)
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Closed trades", len(closed))
            k2.metric("Win rate", f"{win_rate:.1%}")
            k3.metric("Avg win / Avg loss", f"${avg_win:+.2f} / ${avg_loss:+.2f}")
            k4.metric("Profit factor", f"{profit_factor:.2f}")

            # ── Equity curve ──────────────────────────────────────────────────
            st.subheader("Equity curve")
            closed_sorted = closed.sort_values("date")
            closed_sorted["cumulative_pnl"] = closed_sorted["pnl"].cumsum()
            closed_sorted["equity"] = 10_000 + closed_sorted["cumulative_pnl"]
            eq_df = closed_sorted.set_index("date")[["equity"]].copy()
            st.line_chart(eq_df, height=250)

            # ── P&L by ticker ─────────────────────────────────────────────────
            col_ticker, col_reason = st.columns(2)

            with col_ticker:
                st.subheader("P&L by ticker")
                by_ticker = (
                    closed.groupby("ticker")["pnl"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                by_ticker["color"] = by_ticker["pnl"].apply(
                    lambda x: "#2ca02c" if x >= 0 else "#d62728"
                )
                st.dataframe(
                    by_ticker[["ticker", "pnl"]].rename(columns={"pnl": "Total P&L ($)"}),
                    use_container_width=True, hide_index=True,
                )

            with col_reason:
                st.subheader("Exits by reason")
                by_reason = (
                    closed.groupby("reason")["pnl"]
                    .agg(count="count", total="sum", avg="mean")
                    .reset_index()
                    .round(2)
                )
                st.dataframe(by_reason, use_container_width=True, hide_index=True)

            # ── P&L distribution ──────────────────────────────────────────────
            st.subheader("Trade P&L distribution")
            pnl_vals = closed["pnl"].values
            bins = np.linspace(pnl_vals.min(), pnl_vals.max(), 30)
            counts, edges = np.histogram(pnl_vals, bins=bins)
            hist_df = pd.DataFrame({
                "P&L bucket": [f"{e:.1f}" for e in edges[:-1]],
                "count": counts,
            }).set_index("P&L bucket")
            st.bar_chart(hist_df, height=200)

            # ── Raw journal ───────────────────────────────────────────────────
            with st.expander("Raw journal (all rows)"):
                st.dataframe(
                    journal.sort_values("date", ascending=False),
                    use_container_width=True, hide_index=True,
                )

            # ── Open positions ────────────────────────────────────────────────
            if PORTFOLIO_FILE.exists():
                positions = portfolio.get("positions", {})
                if positions:
                    st.subheader("Open positions")
                    pos_rows = []
                    for t, p in positions.items():
                        pos_rows.append({
                            "Ticker": t,
                            "Side": p["side"].upper(),
                            "Entry": f"${p['entry_price']:.2f}",
                            "Shares": round(p["shares"], 4),
                            "Notional": f"${p['notional']:.2f}",
                            "Hold days": p.get("hold_days", 0),
                            "SL": f"{p.get('stop_loss_pct', 0)*100:.1f}%",
                            "TP": f"{p.get('take_profit_pct', 0)*100:.1f}%",
                            "Confidence": f"{p.get('confidence', 0):.1%}",
                        })
                    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
