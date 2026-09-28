"""
Robustness checks: re-run every strategy across estimation windows and
transaction-cost levels (config.ROBUSTNESS_WINDOWS x config.ROBUSTNESS_COSTS_BPS).

A result that only holds for one arbitrary setting is fragile. If the ranking
of strategies survives 1, 3 and 5 year windows and 0-25 bps of costs, we can
trust it more.

Common evaluation period
------------------------
A longer window needs more history, so its first rebalance comes later
(a 5-year window cannot start trading until 5 years after the data begins).
If each window were scored on its own period, differences would mix the
effect of the window with the effect of different market conditions.
So every run is scored on the SAME dates: from the start of the longest
window's out-of-sample period to the end of the data.
"""

import logging

import pandas as pd

import config
from src.backtest import BacktestResult, compute_target_weights, rebalance_dates, simulate
from src.metrics import summarise

logger = logging.getLogger(__name__)


def trim(result: BacktestResult, start: pd.Timestamp) -> BacktestResult:
    """
    Restrict a backtest to dates after `start` (the common evaluation start).

    Turnover and costs keep only rebalances inside the period. The first kept
    rebalance is marked as the "initial" one (turnover excluded from the
    average), matching how the base-case backtest treats its first rebalance.
    """
    keep_r = result.returns.index > start
    keep_rebal = result.turnover.index >= start
    turnover = result.turnover[keep_rebal].copy()
    turnover.iloc[0] = 0.0
    costs = result.costs[keep_rebal].copy()
    costs.iloc[0] = 0.0
    return BacktestResult(
        name=result.name,
        returns=result.returns[keep_r],
        gross_returns=result.gross_returns[keep_r],
        weights=result.weights.loc[result.weights.index >= start],
        turnover=turnover,
        costs=costs,
    )


def run_robustness(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    strategies: dict,
    windows: list[int] = config.ROBUSTNESS_WINDOWS,
    costs_bps: list[float] = config.ROBUSTNESS_COSTS_BPS,
) -> pd.DataFrame:
    """
    Metrics for every (window, cost, strategy) combination, on a common period.

    Weights depend on the window but NOT on the cost level, so each strategy
    is optimised once per window and then simulated at each cost level.

    Returns a long table with columns: window, cost_bps, strategy, and every
    metric from `summarise` (CAGR, Volatility, Sharpe, ...).
    """
    common_start = rebalance_dates(returns.index, max(windows))[0]
    logger.info("Robustness evaluation period starts %s", common_start.date())

    tables = []
    for window in windows:
        weights = {}
        for name, strategy in strategies.items():
            logger.info("Robustness: %s, window=%d", name, window)
            weights[name] = compute_target_weights(returns, risk_free, strategy, window)

        for cost in costs_bps:
            results = {
                name: trim(simulate(returns, w, cost, name), common_start)
                for name, w in weights.items()
            }
            table = summarise(results, risk_free)
            table.insert(0, "strategy", table.index)
            table.insert(0, "cost_bps", cost)
            table.insert(0, "window", window)
            tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    combined.attrs["period"] = f"{common_start:%b %Y} to {returns.index[-1]:%b %Y}"
    return combined
