"""
Run the full Portfolio Construction Shootout pipeline with a single command:

    python main.py

Pipeline (filled in over later milestones):
data -> strategies -> backtest -> metrics -> charts -> robustness checks.
"""

import logging

from src.data import load_data


def main() -> None:
    """Entry point for the full pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # 1. Data: asset returns, benchmark returns and the daily risk-free rate.
    asset_returns, benchmark_returns, risk_free = load_data()
    print(
        f"\nData ready: {asset_returns.shape[1]} assets, {len(asset_returns)} trading days "
        f"({asset_returns.index[0].date()} to {asset_returns.index[-1].date()})."
    )
    print(f"Average annual risk-free rate: {risk_free.mean() * 252:.2%}")


if __name__ == "__main__":
    main()
