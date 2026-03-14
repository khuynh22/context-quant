# Notebooks — Learning Curriculum

These notebooks are a self-contained curriculum that builds up every concept used in ContextQuant.
They do **not** import from `src/` — each one is fully standalone and runs with synthetic data
if real data is unavailable.

Work through the modules in order; each one builds on the last.

```bash
jupyter lab    # launch from the repo root, then open this folder
```

---

## Modules

| Module | Topics |
|--------|--------|
| [`00_pytorch_fundamentals/`](00_pytorch_fundamentals/) | Tensors, autograd, training loops from scratch |
| [`01_time_series_basics/`](01_time_series_basics/) | OHLCV data, technical indicators, normalisation, sliding windows |
| [`02_neural_networks/`](02_neural_networks/) | Linear layers, activations, loss functions, optimisers |
| [`03_lstms_and_sequences/`](03_lstms_and_sequences/) | RNNs, LSTMs, sequence classification |
| [`04_nlp_and_transformers/`](04_nlp_and_transformers/) | Tokenisation, embeddings, FinBERT sentiment pipeline |
| [`05_building_contextquant/`](05_building_contextquant/) | Multimodal data alignment, late-fusion model, end-to-end training |

Each module folder has its own `README.md` listing individual notebooks and their key concepts.
