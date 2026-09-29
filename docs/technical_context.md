# Technical Context: Portfolio Construction Shootout

## Environment
- Python 3.10+
- Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `yfinance`, `pandas-datareader` (for FRED risk-free rate)
- Pin versions in `requirements.txt`.
- Use a virtual environment (`python -m venv .venv`).

## Repository Structure
```
portfolio-construction-shootout/
├── README.md
├── requirements.txt
├── .gitignore            # ignore .venv/, __pycache__/, data/raw/
├── config.py             # all parameters in one place
├── main.py               # runs the full pipeline
├── src/
│   ├── data.py           # download, cache, clean prices; compute returns
│   ├── strategies.py     # one function per strategy, returns weights
│   ├── backtest.py       # rolling rebalance loop, costs, portfolio returns
│   ├── metrics.py        # performance statistics
│   └── plots.py          # all charts
├── tests/
│   └── test_sanity.py    # basic checks (see below)
├── data/
│   └── raw/              # cached downloads (gitignored)
└── output/
    ├── figures/          # PNG charts (commit these, the README uses them)
    └── tables/           # CSV results
```

## Data
- **Universe:** XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB (9 SPDR sector ETFs).
  - XLRE and XLC are excluded because they launched in 2015 and 2018 and would shorten the history.
- **Benchmark:** SPY (reported for comparison only, never in the optimiser).
- **Period:** 2010-01-01 to the latest available date.
- **Prices:** daily adjusted close from yfinance (`auto_adjust=True`).
- **Risk-free rate:** 3-month T-bill (FRED series `DTB3`), converted to a daily rate. If FRED fails, fall back to 0 and print a warning.
- Cache downloads to `data/raw/` as CSV and reload from cache on later runs.
- Use simple daily returns. Drop dates with missing data across the universe and log how many were dropped.

## Configuration (`config.py`)
```python
TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
BENCHMARK = "SPY"
START_DATE = "2010-01-01"
ESTIMATION_WINDOW = 756      # trading days (~3 years)
REBALANCE_FREQ = "M"         # month end
TRANSACTION_COST_BPS = 10    # per unit of turnover
TRADING_DAYS = 252
ROBUSTNESS_WINDOWS = [252, 756, 1260]
ROBUSTNESS_COSTS_BPS = [0, 10, 25]
```

## Backtest Rules (critical)
1. **No look-ahead bias.** At rebalance date *t*, weights are estimated using returns strictly before *t* (window ending at *t−1*). Those weights are applied from *t+1* onwards.
2. **Rebalance** on the last trading day of each month.
3. **Weight drift:** between rebalances, weights drift with asset returns (do not assume constant weights daily).
4. **Turnover** at each rebalance = sum of |new weight − drifted weight|.
5. **Transaction cost** = turnover × cost_bps / 10,000, deducted from the portfolio return on the rebalance day.
6. The backtest starts once the first full estimation window is available, so the out-of-sample period begins around 2013. All strategies share the same out-of-sample dates.

## Strategy Specifications
All strategies: long-only (0 ≤ wᵢ ≤ 1), fully invested (Σwᵢ = 1). Each function takes a returns DataFrame (the estimation window) and returns a pandas Series of weights indexed by ticker.

| Strategy | Method |
|----------|--------|
| Equal weight | wᵢ = 1/N |
| Minimum variance | minimise wᵀΣw, sample covariance, `scipy.optimize.minimize` (SLSQP) |
| Max Sharpe | maximise (wᵀμ − r_f) / √(wᵀΣw), sample mean and covariance, SLSQP |
| Risk parity | equal risk contributions: RCᵢ = wᵢ(Σw)ᵢ / √(wᵀΣw); minimise Σ(RCᵢ − RCⱼ)² or equivalent |
| Min variance (shrinkage) | same as min variance, but Σ from `sklearn.covariance.LedoitWolf` |

- Annualise μ and Σ consistently (×252) or work entirely in daily units. Pick one and comment which.
- If an optimiser fails to converge, fall back to equal weight for that date and log a warning with the date and strategy.
- Start optimisers from equal weight.

## Metrics (`metrics.py`)
For each strategy, over the out-of-sample period, after costs:
- Annualised return (geometric / CAGR)
- Annualised volatility
- Sharpe ratio (excess over risk-free, annualised)
- Maximum drawdown
- Average monthly turnover
- Total transaction costs paid

Output as a DataFrame, saved to `output/tables/results.csv`, and printed to console in a readable format.

## Charts (`plots.py`)
Save as PNG (dpi 150) to `output/figures/`:
1. `cumulative_returns.png`: all strategies plus SPY on one chart
2. `drawdowns.png`: drawdown curves for all strategies
3. `weights_<strategy>.png`: stacked area chart of weights over time, one per strategy
4. `robustness_sharpe.png`: Sharpe by strategy across estimation windows and cost levels

Use clear titles, labelled axes, a legend, and a consistent colour per strategy across all charts.

## Sanity Checks (`tests/test_sanity.py`, run with `pytest`)
- All strategy weights sum to 1 (tolerance 1e-6) and are non-negative.
- Equal-weight returns for one month match a manual calculation.
- No NaNs in the portfolio return series.
- Weights at date *t* do not change if data after *t* is altered (look-ahead test).
- Risk parity risk contributions are approximately equal.

## Code Style
- Type hints and docstrings on every function, explaining the finance concept, not just the code.
- Short comments explaining *why*, especially in the backtest loop and optimisers.
- No hard-coded parameters outside `config.py`.
- Use `print` or `logging` for progress so it's clear what the run is doing.

## Git Workflow
- Commit after each working milestone with a clear message (e.g. "Add minimum variance strategy").
- Push to GitHub `main` after each commit.
- Never commit `.venv/`, `__pycache__/`, or `data/raw/`.
- Do commit `output/figures/` so the README images render on GitHub.
