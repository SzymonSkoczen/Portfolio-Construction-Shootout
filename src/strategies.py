"""
Portfolio construction rules.

Every strategy has the same signature:

    strategy(returns, rf=0.0) -> pd.Series of weights indexed by ticker

where `returns` is the estimation window of daily asset returns (only data
available *before* the rebalance date) and `rf` is the annualised risk-free
rate over that window (only max Sharpe uses it).

All strategies are long-only (0 <= w_i <= 1) and fully invested (sum w_i = 1),
so no short selling and no leverage.

Units: all inputs to the optimisers are ANNUALISED. Expected returns are daily
means x 252 and covariances are daily covariances x 252. Using annual units
keeps numbers readable (e.g. 18% volatility, not 1.1% daily). Weights are the
same either way because scaling mu and Sigma by the same factor does not
change which portfolio is optimal.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config

logger = logging.getLogger(__name__)


# --- Estimation helpers -----------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Annualised sample covariance matrix.

    With 9 assets there are 45 distinct covariances to estimate from a few
    years of data, so each one is noisy. Optimisers treat these noisy numbers
    as exact, which is the root of the "estimation error" problem.
    """
    return returns.cov().to_numpy() * config.TRADING_DAYS


def shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Annualised Ledoit-Wolf shrinkage covariance matrix.

    Ledoit-Wolf blends the noisy sample covariance S with a simple, very stable
    target F (a scaled identity matrix: every asset gets the same variance and
    zero correlation):
        Sigma_shrunk = delta * F + (1 - delta) * S
    The shrinkage intensity delta is chosen from the data to minimise expected
    estimation error. Pulling extreme covariances towards the average stops
    the optimiser from betting heavily on correlations that are just noise.
    """
    lw = LedoitWolf().fit(returns.to_numpy())
    return lw.covariance_ * config.TRADING_DAYS


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Each asset's contribution to portfolio volatility.

        RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w)

    (Sigma w)_i is how much portfolio variance rises if we add a little of
    asset i (its marginal risk). Multiplying by w_i gives the asset's share of
    the total. The contributions add up exactly to portfolio volatility.
    """
    weights = np.asarray(weights)
    port_vol = np.sqrt(weights @ cov @ weights)
    return weights * (cov @ weights) / port_vol


# --- Optimiser wrapper ------------------------------------------------------

def _optimise(
    objective: Callable[[np.ndarray], float],
    tickers: pd.Index,
    strategy_name: str,
    as_of: pd.Timestamp,
) -> pd.Series:
    """
    Minimise `objective` over long-only, fully invested weights using SLSQP.

    - Starts from equal weight, a neutral and always-feasible point.
    - Bounds 0 <= w_i <= 1 give long-only weights, and the equality
      constraint sum(w) = 1 means the portfolio is fully invested.
    - If the solver fails, returns equal weight and logs a warning, so a
      single bad month cannot crash the whole backtest.
    """
    n = len(tickers)
    x0 = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        logger.warning(
            "%s optimiser failed on %s (%s); falling back to equal weight.",
            strategy_name, as_of.date(), result.message,
        )
        return pd.Series(x0, index=tickers)

    # SLSQP can return tiny negative values like -1e-17; clean them up and
    # re-normalise so weights are exactly non-negative and sum to one.
    w = np.clip(result.x, 0.0, None)
    return pd.Series(w / w.sum(), index=tickers)


# --- Strategies -------------------------------------------------------------

def equal_weight(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Naive diversification: w_i = 1/N.

    Uses no estimates at all, so it has zero estimation error. That is the
    whole point of DeMiguel et al. (2009): anything cleverer must earn back
    the error it introduces by estimating means and covariances.
    """
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


def min_variance(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Global minimum variance: minimise w' Sigma w (sample covariance).

    Ignores expected returns entirely. Means are much harder to estimate than
    covariances, so dropping them removes the noisiest input. It tends to
    overweight low-volatility, low-correlation sectors such as utilities and
    consumer staples.
    """
    cov = sample_covariance(returns)
    return _optimise(lambda w: w @ cov @ w, returns.columns, "min_variance", returns.index[-1])


def min_variance_shrinkage(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Minimum variance using the Ledoit-Wolf shrinkage covariance.

    Same objective as `min_variance`; only the covariance estimate changes.
    Comparing the two isolates whether a better covariance estimate improves
    out-of-sample results and stabilises weights (lower turnover).
    """
    cov = shrunk_covariance(returns)
    return _optimise(lambda w: w @ cov @ w, returns.columns, "min_variance_shrinkage", returns.index[-1])


def max_sharpe(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Tangency (mean-variance) portfolio: maximise (w' mu - rf) / sqrt(w' Sigma w).

    Uses sample mean returns, which are extremely noisy: a sector that simply
    had a lucky few years looks like a great investment. The optimiser then
    concentrates in it, so weights are unstable and turnover is high. We
    minimise the negative Sharpe ratio because scipy only minimises.
    """
    mu = returns.mean().to_numpy() * config.TRADING_DAYS
    cov = sample_covariance(returns)

    def negative_sharpe(w: np.ndarray) -> float:
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    return _optimise(negative_sharpe, returns.columns, "max_sharpe", returns.index[-1])


def risk_parity(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Equal risk contribution: every asset contributes the same share of risk.

    Equal weight equalises *capital*, but a volatile sector (e.g. energy) then
    dominates portfolio risk. Risk parity equalises *risk* instead, giving
    less capital to volatile assets. Like min variance it needs no expected
    returns, but unlike min variance it never drops an asset completely.

    We minimise sum_i (RC_i / sigma_p - 1/N)^2, i.e. the gap between each
    asset's percentage risk share and the equal share 1/N. This is
    equivalent to minimising sum_ij (RC_i - RC_j)^2 but better scaled for
    the solver, because percentage shares are numbers of order 0.1.
    """
    cov = sample_covariance(returns)
    n = returns.shape[1]

    def risk_share_gap(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(np.sum((rc / rc.sum() - 1.0 / n) ** 2))

    return _optimise(risk_share_gap, returns.columns, "risk_parity", returns.index[-1])


# Registry used by the backtest, metrics and plots. The order sets the order of
# tables and legends; names are the labels shown in outputs.
STRATEGIES: dict[str, Callable[[pd.DataFrame, float], pd.Series]] = {
    "Equal Weight": equal_weight,
    "Min Variance": min_variance,
    "Max Sharpe": max_sharpe,
    "Risk Parity": risk_parity,
    "Min Var (Shrinkage)": min_variance_shrinkage,
}
