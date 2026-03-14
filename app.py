"""
app.py — ContextQuant Streamlit Dashboard.

Run with:
    streamlit run app.py

Features
--------
- Select ticker + date range in the sidebar
- Optionally enter a news headline for live FinBERT sentiment
- One-click "Analyse" button: downloads data, runs inference
- 5-class probability bar chart + guidance banner
- Raw OHLCV + indicator preview table
- Load / train model controls
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data_loader import (
    CLASS_LABELS,
    FEATURE_COLS,
    add_technical_indicators,
    download_ohlcv,
)
from src.model import ContextQuantFusionNet, build_model
from src.utils import set_seed

MODELS_DIR: Path = Path(__file__).parent / "models"

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextQuant",
    page_icon="📈",
    layout="wide",
)

# ── Colour map for guidance classes ─────────────────────────────────────────
CLASS_COLORS = {
    "Strong Sell": "#d62728",
    "Sell":        "#ff7f0e",
    "Hold":        "#bcbd22",
    "Buy":         "#2ca02c",
    "Strong Buy":  "#1f77b4",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model(checkpoint_path: Path) -> ContextQuantFusionNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_data(show_spinner="Downloading market data …")
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = download_ohlcv(ticker, start=start, end=end)
        df = add_technical_indicators(df)
        return df.dropna()
    except Exception as exc:
        st.error(f"Could not fetch data: {exc}")
        return None


def sentiment_from_finbert(headline: str) -> torch.Tensor:
    """Return a [1, 3] tensor of [positive, negative, neutral] probs.

    Falls back to rule-based scoring if transformers is unavailable.
    """
    try:
        from transformers import pipeline  # type: ignore
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        results = clf(headline)[0]
        label_map = {"positive": 0, "negative": 1, "neutral": 2}
        probs = [0.0, 0.0, 0.0]
        for r in results:
            idx = label_map.get(r["label"].lower(), 2)
            probs[idx] = r["score"]
    except Exception:
        # Rule-based fallback
        positive_words = {"strong", "growth", "raises", "profit", "beat", "up", "rally"}
        negative_words = {"concern", "weigh", "loss", "cut", "down", "fall", "miss"}
        t = headline.lower()
        p = sum(w in t for w in positive_words)
        n = sum(w in t for w in negative_words)
        if p > n:
            probs = [0.7, 0.1, 0.2]
        elif n > p:
            probs = [0.1, 0.7, 0.2]
        else:
            probs = [1 / 3, 1 / 3, 1 / 3]
    return torch.tensor([probs], dtype=torch.float32)


def build_inference_window(df: pd.DataFrame, window_size: int = 60) -> torch.Tensor | None:
    """Take the last *window_size* rows and return a normalised [1, 60, 11] tensor."""
    from sklearn.preprocessing import MinMaxScaler

    if len(df) < window_size:
        st.warning(f"Need at least {window_size} rows of data; got {len(df)}.")
        return None
    window = df[FEATURE_COLS].iloc[-window_size:].values.astype(np.float32)
    scaler = MinMaxScaler()
    window_scaled = scaler.fit_transform(window)
    return torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)  # [1, 60, 11]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ContextQuant")
    st.caption("Multi-modal stock guidance · 5-class signal")
    st.divider()

    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    start_date = col1.text_input("Start", value="2022-01-01")
    end_date = col2.text_input("End", value="2025-12-31")

    st.divider()
    headline = st.text_area(
        "News headline (optional)",
        placeholder="e.g. 'Company beats earnings expectations'",
        height=80,
    )

    st.divider()
    st.subheader("Model")
    checkpoint_files = sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    checkpoint_options = [p.name for p in checkpoint_files]
    selected_ckpt = st.selectbox(
        "Checkpoint",
        options=["(none — train first)"] + checkpoint_options,
    )

    train_btn = st.button("Train model now", use_container_width=True)
    analyse_btn = st.button("Analyse", type="primary", use_container_width=True)


# ── Train on demand ──────────────────────────────────────────────────────────

if train_btn:
    with st.spinner(f"Training on {ticker} …"):
        set_seed(42)
        try:
            from src.train import train as run_training

            history = run_training(
                ticker=ticker,
                n_epochs=30,
                lr=1e-3,
                batch_size=64,
                patience=8,
            )
            final_acc = history["val_acc"][-1]
            st.success(f"Training complete — val_acc={final_acc:.3f}")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")


# ── Main panel ────────────────────────────────────────────────────────────────

st.title("📈 ContextQuant — Stock Guidance Dashboard")

if not analyse_btn:
    st.info("Configure a ticker in the sidebar, then click **Analyse**.")
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
df = get_data(ticker, start_date, end_date)
if df is None or df.empty:
    st.stop()

# ── Build temporal window ────────────────────────────────────────────────────
x_temporal = build_inference_window(df)
if x_temporal is None:
    st.stop()

# ── Sentiment vector ─────────────────────────────────────────────────────────
if headline.strip():
    x_sentiment = sentiment_from_finbert(headline.strip())
    st.caption(
        f"Sentiment probs — positive={x_sentiment[0, 0]:.3f}  "
        f"negative={x_sentiment[0, 1]:.3f}  neutral={x_sentiment[0, 2]:.3f}"
    )
else:
    x_sentiment = None   # model will use neutral prior

# ── Load / create model ───────────────────────────────────────────────────────
if selected_ckpt != "(none — train first)" and selected_ckpt:
    ckpt_path = MODELS_DIR / selected_ckpt
    try:
        model = load_model(ckpt_path)
        st.caption(f"Loaded checkpoint: `{selected_ckpt}`")
    except Exception as exc:
        st.warning(f"Could not load checkpoint ({exc}). Using untrained model.")
        model = build_model()
        model.eval()
else:
    model = build_model()
    model.eval()
    st.warning("No trained checkpoint selected — probabilities are from an untrained model.")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    probs = model.predict_proba(x_temporal, x_sentiment).squeeze().tolist()

predicted_idx = int(np.argmax(probs))
predicted_label = CLASS_LABELS[predicted_idx]

# ── Guidance banner ───────────────────────────────────────────────────────────
banner_color = CLASS_COLORS[predicted_label]
st.markdown(
    f"""
    <div style="background:{banner_color}22;border-left:6px solid {banner_color};
                padding:1rem 1.5rem;border-radius:4px;margin-bottom:1.5rem">
        <span style="font-size:2rem;font-weight:700;color:{banner_color}">{predicted_label}</span>
        <span style="margin-left:1rem;font-size:1.1rem;color:#666">
            {ticker} · {df.index[-1].date()}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Probability chart ─────────────────────────────────────────────────────────
col_chart, col_tbl = st.columns([3, 2])

with col_chart:
    st.subheader("Class probabilities")
    prob_df = pd.DataFrame(
        {"Class": CLASS_LABELS, "Probability": probs}
    ).set_index("Class")
    st.bar_chart(prob_df, color="#1f77b4", height=280)

with col_tbl:
    st.subheader("Probability table")
    display_df = pd.DataFrame(
        {
            "Class": CLASS_LABELS,
            "Probability": [f"{p:.1%}" for p in probs],
        }
    )
    display_df["Selected"] = ["⬅" if i == predicted_idx else "" for i in range(5)]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Recent OHLCV preview ──────────────────────────────────────────────────────
with st.expander("Recent market data (last 10 rows)"):
    preview_cols = ["Open", "High", "Low", "Close", "Volume",
                    "RSI_14", "MACD", "BB_HIGH", "BB_LOW"]
    available = [c for c in preview_cols if c in df.columns]
    st.dataframe(df[available].tail(10).round(4), use_container_width=True)
