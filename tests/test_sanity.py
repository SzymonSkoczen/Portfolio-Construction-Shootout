"""
Sanity checks for the strategies and the backtest (run with `pytest`).

The tests use synthetic returns with a fixed random seed, so they run offline,
quickly, and give the same answer every time. The synthetic data has a common
"market" factor so assets are correlated, like real sector ETFs.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import compute_target_weights, rebalance_dates, simulate
from src.strategies import STRATEGIES, risk_contributions, sample_covariance

TICKERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
WINDOW = 60      # short window keeps tests fast; the logic is identical to 756
COST_BPS = 10


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """~2 years of daily returns: market factor + asset-specific noise."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    market = rng.normal(0.0004, 0.01, size=(len(dates), 1))
    betas = np.linspace(0.6, 1.4, len(TICKERS))          # different sensitivities
    idio_vol = np.linspace(0.005, 0.015, len(TICKERS))   # different specific risk
    noise = rng.normal(0.0, 1.0, size=(len(dates), len(TICKERS))) * idio_vol
    return pd.DataFrame(market * betas + noise, index=dates, columns=TICKERS)


@pytest.fixture(scope="module")
def risk_free(returns: pd.DataFrame) -> pd.Series:
    """Constant 2% annual risk-free rate, converted to daily."""
    return pd.Series((1.02) ** (1 / 252) - 1, index=returns.index)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_weights_sum_to_one_and_non_negative(name, returns, risk_free):
    """Long-only, fully invested: every weight >= 0 and weights sum to 1."""
    weights = compute_target_weights(returns, risk_free, STRATEGIES[name], WINDOW)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert (weights >= 0).all().all()


def test_equal_weight_month_matches_manual(returns, risk_free):
    """
    Over one month, 1/N with drift is buy-and-hold: invest 1/N in each asset
    at the start, do nothing, and the portfolio's growth is the average of
    the assets' growth. The backtest's gross return must match that.
    """
    weights = compute_target_weights(returns, risk_free, STRATEGIES["Equal Weight"], WINDOW)
    result = simulate(returns, weights, cost_bps=0)

    start, end = weights.index[1], weights.index[2]
    month = returns.loc[(returns.index > start) & (returns.index <= end)]
    manual = (1 + month).prod().mean() - 1
    backtest = (1 + result.gross_returns.loc[month.index]).prod() - 1
    assert backtest == pytest.approx(manual, abs=1e-12)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_no_nans_in_portfolio_returns(name, returns, risk_free):
    """Every out-of-sample day must have a valid return, before and after costs."""
    weights = compute_target_weights(returns, risk_free, STRATEGIES[name], WINDOW)
    result = simulate(returns, weights, COST_BPS)
    assert not result.returns.isna().any()
    assert not result.gross_returns.isna().any()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_no_look_ahead(name, returns, risk_free):
    """
    Weights chosen at date t may only use data before t. If we scramble every
    return from t onwards, the weights at t (and all earlier dates) must not
    change at all.
    """
    dates = rebalance_dates(returns.index, WINDOW)
    t = dates[len(dates) // 2]

    altered = returns.copy()
    rng = np.random.default_rng(0)
    future = altered.index >= t
    altered.loc[future] = rng.normal(0.0, 0.05, size=(future.sum(), len(TICKERS)))

    original = compute_target_weights(returns, risk_free, STRATEGIES[name], WINDOW)
    changed = compute_target_weights(altered, risk_free, STRATEGIES[name], WINDOW)
    pd.testing.assert_frame_equal(original.loc[:t], changed.loc[:t])


def test_risk_parity_contributions_equal(returns, risk_free):
    """Under risk parity each asset contributes ~1/N of portfolio volatility."""
    window = returns.iloc[-WINDOW:]
    weights = STRATEGIES["Risk Parity"](window, 0.0)
    rc = risk_contributions(weights.to_numpy(), sample_covariance(window))
    shares = rc / rc.sum()
    assert np.allclose(shares, 1 / len(TICKERS), atol=1e-4)


def test_costs_equal_turnover_times_bps(returns, risk_free):
    """Each rebalance's cost is turnover x bps / 10,000, and net = gross - cost."""
    weights = compute_target_weights(returns, risk_free, STRATEGIES["Min Variance"], WINDOW)
    result = simulate(returns, weights, COST_BPS)
    assert np.allclose(result.costs, result.turnover * COST_BPS / 10_000)
    assert result.turnover.iloc[0] == 0.0  # initial purchase is not charged

    rebal_days = result.costs.index[1:]
    diff = result.gross_returns.loc[rebal_days] - result.returns.loc[rebal_days]
    assert np.allclose(diff, result.costs.loc[rebal_days])
