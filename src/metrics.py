"""
Out-of-sample performance statistics, all computed on returns AFTER costs.

Conventions
-----------
- Daily returns are simple returns; a year has config.TRADING_DAYS (252) days.
- Returns are annualised geometrically (CAGR) because that is the growth rate
  an investor actually experienced. Volatility scales with sqrt(252) because
  variance of independent daily returns adds up over time.
"""

import numpy as np
import pandas as pd

import config
from src.backtest import BacktestResult


def cagr(returns: pd.Series, trading_days: int = config.TRADING_DAYS) -> float:
    """
    Compound annual growth rate: the constant yearly return that turns $1
    into the same final wealth.

        CAGR = (prod(1 + r_t))^(trading_days / n_days) - 1
    """
    growth = (1 + returns).prod()
    years = len(returns) / trading_days
    return growth ** (1 / years) - 1


def annualised_volatility(returns: pd.Series, trading_days: int = config.TRADING_DAYS) -> float:
    """Standard deviation of daily returns scaled to a year: sigma_daily * sqrt(252)."""
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(
    returns: pd.Series, risk_free: pd.Series, trading_days: int = config.TRADING_DAYS
) -> float:
    """
    Annualised Sharpe ratio: average excess return per unit of volatility.

        Sharpe = mean(r_t - rf_t) / std(r_t - rf_t) * sqrt(252)

    Subtracting the T-bill rate day by day measures the reward for taking
    risk, rather than for simply holding cash. This matters here because rates
    moved from ~0% (2013-2021) to ~5% (2023-2024).
    """
    excess = returns - risk_free.reindex(returns.index)
    return excess.mean() / excess.std() * np.sqrt(trading_days)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Percentage fall from the running peak of wealth at each date.

    A value of -0.25 means the portfolio is 25% below its previous high.
    """
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss over the whole period (a negative number)."""
    return drawdown_series(returns).min()


def summarise(
    results: dict[str, BacktestResult],
    risk_free: pd.Series,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """
    One row per strategy with every headline metric.

    - Avg monthly turnover excludes the initial purchase (which is 0 by
      construction in the backtest) so it measures ongoing trading only.
    - Total costs is the sum of all costs paid, expressed as a fraction of
      portfolio value (0.05 = 5 percentage points of return given up).

    If a benchmark return series is given (SPY), it is added as a final row
    over the same dates. It has no turnover or costs because we do not trade it.
    """
    rows = {}
    for name, res in results.items():
        r = res.returns
        rows[name] = {
            "CAGR": cagr(r),
            "Volatility": annualised_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free),
            "Max Drawdown": max_drawdown(r),
            "Avg Monthly Turnover": res.turnover.iloc[1:].mean(),
            "Total Costs": res.costs.sum(),
        }

    if benchmark is not None:
        dates = next(iter(results.values())).returns.index
        b = benchmark.reindex(dates)
        rows[benchmark.name or "Benchmark"] = {
            "CAGR": cagr(b),
            "Volatility": annualised_volatility(b),
            "Sharpe": sharpe_ratio(b, risk_free),
            "Max Drawdown": max_drawdown(b),
            "Avg Monthly Turnover": np.nan,
            "Total Costs": np.nan,
        }

    return pd.DataFrame(rows).T


def format_table(table: pd.DataFrame) -> str:
    """Readable console version: percentages for returns/risk/costs, 2 d.p. for Sharpe."""
    formatted = table.copy().astype(object)
    for col in table.columns:
        if col == "Sharpe":
            formatted[col] = table[col].map(lambda x: f"{x:.2f}")
        else:
            formatted[col] = table[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    return formatted.to_string()
