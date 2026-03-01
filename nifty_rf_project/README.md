# Nifty RF Project

A modular Python implementation of a **Random Forest-based Nifty 5-minute strategy** with:

- model training
- backtesting engine
- paper trading mode (no live execution)
- Streamlit analytics dashboard

## Project Structure

```text
nifty_rf_project/
├── config.py
├── train_model.py
├── backtest.py
├── paper_trade.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Strategy Logic

### Entry filter
- Daily VWAP reset and intraday VWAP tracking
- Price trigger: current close must be at least **-0.5% from previous close**
- ML filter: model probability must be above selected threshold

### Exit logic
- Target: **+0.6%**
- Stop loss: **-0.4%**
- End-of-day forced square-off

### Risk analytics
- Equity curve
- Drawdown curve
- Rolling performance / rolling Sharpe
- Risk-adjusted metrics (Sharpe, Sortino, CAGR proxy, volatility)

---

## Data Format

Input CSV expected columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Place your data at `nifty_rf_project/data/nifty_5min.csv` (or pass custom path via `--data`).

## How to Install

```bash
cd nifty_rf_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Train Model

```bash
cd nifty_rf_project
python train_model.py --data data/nifty_5min.csv
```

Model artifact is saved to:

- `nifty_rf_project/models/nifty_rf_model.joblib`

## How to Run Backtest

```bash
cd nifty_rf_project
python backtest.py --data data/nifty_5min.csv --threshold 0.60 --capital 1000000
```

## How to Run Paper Trading

```bash
cd nifty_rf_project
python paper_trade.py --data data/nifty_5min.csv --threshold 0.60
```

This only generates simulated signals and does not place real orders.

## How to Run Dashboard

```bash
cd nifty_rf_project
streamlit run dashboard.py
```

Dashboard includes:
- threshold slider
- 30/45/60 day selection
- equity curve
- drawdown curve
- rolling performance
- risk-adjusted metrics
