"""
Run the full Portfolio Construction Shootout pipeline with a single command:

    python main.py

Pipeline (filled in over later milestones):
data -> strategies -> backtest -> metrics -> charts -> robustness checks.
"""

import logging

import config
from src.backtest import run_backtests
from src.data import load_data
from src.metrics import format_table, summarise
from src.plots import plot_all
from src.strategies import STRATEGIES


def main() -> None:
    """Entry point for the full pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # 1. Data: asset returns, benchmark returns and the daily risk-free rate.
    asset_returns, benchmark_returns, risk_free = load_data()
    print(
        f"\nData ready: {asset_returns.shape[1]} assets, {len(asset_returns)} trading days "
        f"({asset_returns.index[0].date()} to {asset_returns.index[-1].date()})."
    )

    # 2. Backtest every strategy with the base-case window and cost level.
    results = run_backtests(asset_returns, risk_free, STRATEGIES)
    oos = next(iter(results.values())).returns.index
    print(f"Out-of-sample period: {oos[0].date()} to {oos[-1].date()} ({len(oos)} days).")

    # 3. Metrics table (after costs), printed and saved.
    table = summarise(results, risk_free, benchmark_returns)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "results.csv", float_format="%.6f")
    print(
        f"\nResults after costs (window={config.ESTIMATION_WINDOW} days, "
        f"cost={config.TRANSACTION_COST_BPS} bps):\n"
    )
    print(format_table(table))

    # 4. Charts for the base case.
    plot_all(results, benchmark_returns)
    print(f"\nCharts saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
