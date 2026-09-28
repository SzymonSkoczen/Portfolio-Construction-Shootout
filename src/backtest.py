"""
Rolling out-of-sample backtest with monthly rebalancing, weight drift and
transaction costs.

The backtest has two stages, kept separate on purpose:

1. `compute_target_weights`: at each month end, ask the strategy for new
   weights using ONLY past data. This is where look-ahead bias is avoided.
2. `simulate`: hold those weights through time, let them drift with market
   moves, and charge costs whenever we trade back to target.

Separating them means the (slow) optimisation runs once per estimation window,
and the robustness checks can re-simulate with different cost levels for free.

Timeline at a rebalance date t (the last trading day of a month):

    ... t-window ... t-1 | t          | t+1 ...
    [ estimation data  ] | rebalance  | new weights earn returns
                         | at close   |

- Weights use returns from t-window to t-1 (strictly before t).
- During day t the portfolio still holds the OLD (drifted) weights and earns
  day t's return with them.
- At the close of t we trade to the new weights and pay the cost, which is
  deducted from day t's return.
- From t+1 onwards the new weights earn returns.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

Strategy = Callable[[pd.DataFrame, float], pd.Series]


@dataclass
class BacktestResult:
    """Everything one strategy's backtest produces."""

    name: str
    returns: pd.Series          # daily portfolio returns AFTER costs
    gross_returns: pd.Series    # daily portfolio returns BEFORE costs
    weights: pd.DataFrame       # target weights set at each rebalance date
    turnover: pd.Series         # turnover at each rebalance date
    costs: pd.Series            # cost (as a return) paid at each rebalance date


def rebalance_dates(index: pd.DatetimeIndex, window: int) -> pd.DatetimeIndex:
    """
    Month-end rebalance dates that have a full estimation window behind them.

    A date is a month end if the next trading day falls in a different month.
    The final date in the data is excluded because we cannot tell whether it
    is a month end (the month may not be over yet).

    The first usable date needs at least `window` returns strictly before it,
    i.e. its position in the index must be >= window. Because this depends
    only on the dates and the window, every strategy gets the same rebalance
    dates and therefore the same out-of-sample period.
    """
    months = index.to_period("M")
    is_month_end = months[:-1] != months[1:]
    positions = np.flatnonzero(is_month_end)
    positions = positions[positions >= window]
    return index[positions]


def compute_target_weights(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    strategy: Strategy,
    window: int = config.ESTIMATION_WINDOW,
) -> pd.DataFrame:
    """
    Target weights at every rebalance date, estimated without look-ahead.

    For rebalance date t at position p, the estimation window is rows
    p-window to p-1 of `returns`: the `window` trading days ending the day
    BEFORE t. Nothing from day t or later is visible to the strategy.

    The risk-free rate passed to the strategy is the average daily rate over
    the same window, annualised (only max Sharpe uses it).
    """
    weights = {}
    for date in rebalance_dates(returns.index, window):
        p = returns.index.get_loc(date)
        estimation_window = returns.iloc[p - window : p]
        rf_annual = risk_free.iloc[p - window : p].mean() * config.TRADING_DAYS
        weights[date] = strategy(estimation_window, rf_annual)
    return pd.DataFrame(weights).T


def simulate(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = config.TRANSACTION_COST_BPS,
    name: str = "",
) -> BacktestResult:
    """
    Simulate daily portfolio returns from a schedule of target weights.

    Weight drift: between rebalances we do not trade, so each asset's weight
    changes with its own performance:
        w_i,new = w_i * (1 + r_i) / (1 + r_portfolio)
    Winners grow as a share of the portfolio and losers shrink. Assuming
    constant weights would secretly rebalance every day for free.

    Turnover at a rebalance = sum_i |target_i - drifted_i|, the fraction of
    the portfolio traded (buys plus sells). Cost = turnover x cost_bps / 10,000,
    subtracted from that day's return.

    The initial purchase at the first rebalance is not charged. Every strategy
    would pay the same one-off cost to enter the market, and leaving it out
    keeps average turnover a pure measure of ongoing trading.

    The out-of-sample return series starts the day after the first rebalance.
    """
    first = target_weights.index[0]
    start = returns.index.get_loc(first)
    asset_returns = returns.to_numpy()
    dates = returns.index
    tickers = returns.columns

    # Look up target weights by index position, aligned to the return columns.
    targets = {
        dates.get_loc(d): target_weights.loc[d, tickers].to_numpy()
        for d in target_weights.index
    }

    w = targets[start].copy()                   # invested at the close of the first rebalance date
    gross = np.zeros(len(dates) - start - 1)
    net = np.zeros_like(gross)
    turnover, costs = {first: 0.0}, {first: 0.0}

    for p in range(start + 1, len(dates)):
        r = asset_returns[p]

        # 1. Earn today's return with the weights held from yesterday's close.
        port_r = float(w @ r)

        # 2. Let the weights drift with today's moves (no trading).
        w = w * (1 + r) / (1 + port_r)

        # 3. At a month-end close, trade back to the new targets and pay costs.
        cost = 0.0
        if p in targets:
            new_w = targets[p]
            trade = float(np.abs(new_w - w).sum())
            cost = trade * cost_bps / 10_000
            turnover[dates[p]] = trade
            costs[dates[p]] = cost
            w = new_w.copy()

        gross[p - start - 1] = port_r
        net[p - start - 1] = port_r - cost

    oos_dates = dates[start + 1 :]
    return BacktestResult(
        name=name,
        returns=pd.Series(net, index=oos_dates, name=name),
        gross_returns=pd.Series(gross, index=oos_dates, name=name),
        weights=target_weights,
        turnover=pd.Series(turnover, name=name),
        costs=pd.Series(costs, name=name),
    )


def run_backtests(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    strategies: dict[str, Strategy],
    window: int = config.ESTIMATION_WINDOW,
    cost_bps: float = config.TRANSACTION_COST_BPS,
) -> dict[str, BacktestResult]:
    """Run every strategy with the same data, window and cost level."""
    results = {}
    for name, strategy in strategies.items():
        logger.info("Backtesting %s (window=%d, cost=%s bps)...", name, window, cost_bps)
        weights = compute_target_weights(returns, risk_free, strategy, window)
        results[name] = simulate(returns, weights, cost_bps, name)
    return results
