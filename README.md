# Portfolio Construction Shootout

> Does portfolio optimisation actually beat a simple equal-weight portfolio out of sample, once realistic
> constraints and transaction costs are included, and if not, why not?

This project compares five long-only strategies on nine SPDR sector ETFs (2010 to today), after transaction
costs: equal weight (1/N), minimum variance, max Sharpe, risk parity, and minimum variance with Ledoit-Wolf
shrinkage. SPY is shown as an external benchmark. It builds on DeMiguel, Garlappi & Uppal (2009).

*Work in progress. Results, charts and findings will be added once the pipeline is complete.*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                   # runs the full pipeline
pytest                           # runs the sanity checks
```

## Project status

- [x] Project structure and configuration (`config.py`)
- [ ] Data pipeline (yfinance prices, FRED risk-free rate, caching)
- [ ] Strategies
- [ ] Backtest engine (no look-ahead, weight drift, costs)
- [ ] Sanity tests
- [ ] Metrics
- [ ] Charts
- [ ] Robustness checks
- [ ] Final write-up

See `project.context.md` and `technical_context.md` for the full specification.
