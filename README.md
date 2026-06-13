# Optimal Transport for Regime‑Warped Time Series

Applies optimal transport (Wasserstein distance) to align ETF return distributions across macro regimes. The score is the slope of the transport map – how much the expected return shifts when macro (e.g., VIX) changes.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Macro‑dependent warping of return distribution
- Score = warped expected return − current mean return
- Positive score → macro suggests upward shift
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-optimal-transport-warped-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive score → ETF expected to rise due to macro shift.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
