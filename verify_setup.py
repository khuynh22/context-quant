"""
Run this after setting up the uv environment to confirm everything is working.

    uv run python verify_setup.py
"""

import sys

checks = []

# ── Python version ────────────────────────────────────────────────────────────
major, minor = sys.version_info[:2]
ok = major == 3 and minor >= 11
checks.append(("Python 3.11+", ok, f"{major}.{minor}"))

# ── Core libraries ────────────────────────────────────────────────────────────
libs = [
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("matplotlib",   "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("tqdm",         "tqdm"),
]
for label, module in libs:
    try:
        m = __import__(module)
        checks.append((label, True, getattr(m, "__version__", "ok")))
    except ImportError:
        checks.append((label, False, "NOT FOUND"))

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_str = "YES (" + torch.cuda.get_device_name(0) + ")" if cuda_available else "no (CPU only)"
    device_info = f"{torch.__version__}  |  CUDA: {cuda_str}"
    # Quick tensor smoke-test
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert (a + b).sum().item() == 21.0
    checks.append(("PyTorch", True, device_info))
except ImportError:
    checks.append(("PyTorch", False, "NOT FOUND"))
except Exception as e:
    checks.append(("PyTorch", False, str(e)))

# ── Hugging Face Transformers ─────────────────────────────────────────────────
try:
    import transformers
    checks.append(("transformers", True, transformers.__version__))
except ImportError:
    checks.append(("transformers", False, "NOT FOUND"))

# ── Finance libs ──────────────────────────────────────────────────────────────
try:
    import yfinance
    checks.append(("yfinance", True, yfinance.__version__))
except ImportError:
    checks.append(("yfinance", False, "NOT FOUND"))

try:
    import ta  # noqa: F401
    checks.append(("ta (indicators)", True, getattr(ta, "__version__", "ok")))
except ImportError:
    checks.append(("ta (indicators)", False, "NOT FOUND"))

# ── Streamlit ─────────────────────────────────────────────────────────────────
try:
    import streamlit
    checks.append(("streamlit", True, streamlit.__version__))
except ImportError:
    checks.append(("streamlit", False, "NOT FOUND"))

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ContextQuant — Environment Verification")
print("=" * 60)
all_ok = True
for name, ok, detail in checks:
    status = "✓" if ok else "✗"
    print(f"  {status}  {name:<20}  {detail}")
    if not ok:
        all_ok = False

print("=" * 60)
if all_ok:
    print("  All checks passed. You're ready to open Jupyter Lab!\n")
else:
    print("  Some checks failed. Run:  uv sync\n")
