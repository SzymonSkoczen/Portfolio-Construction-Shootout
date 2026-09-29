"""
Is a strategy's Sharpe ratio *significantly* different from equal weight's?

A backtest only gives one realised history. Two strategies can show different
Sharpe ratios purely by chance, so we test the null hypothesis

    H0: Sharpe(strategy) = Sharpe(equal weight)

with two complementary methods, both on daily EXCESS returns (return minus
the T-bill rate) over the same out-of-sample dates:

1. Jobson-Korkie test with Memmel's (2003) correction. A closed-form z-test.
   It accounts for the two strategies being highly correlated (they hold the
   same assets), which makes differences easier to detect than if they were
   independent. It assumes returns are i.i.d. normal, which daily returns
   are not (fat tails, volatility clustering).

2. Circular block bootstrap (in the spirit of Ledoit & Wolf, 2008). We
   resample blocks of consecutive days, keeping both strategies' returns on
   the same days together, and recompute the Sharpe difference thousands of
   times. Blocks preserve volatility clustering, so no normality or
   independence assumption is needed. This is the more robust of the two.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from src.backtest import BacktestResult


def _excess(returns: pd.Series, risk_free: pd.Series) -> np.ndarray:
    """Daily excess returns as a numpy array."""
    return (returns - risk_free.reindex(returns.index)).to_numpy()


def _sharpe(x: np.ndarray) -> float:
    """Daily (not annualised) Sharpe ratio of an excess return series."""
    return x.mean() / x.std(ddof=1)


def memmel_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Jobson-Korkie / Memmel test of equal Sharpe ratios.

    With daily Sharpe ratios SR_a, SR_b, correlation rho and T observations:
        V = 2 - 2 rho + 0.5 (SR_a^2 + SR_b^2 - 2 SR_a SR_b rho^2)
        z = (SR_a - SR_b) / sqrt(V / T)
    Under H0, z is approximately standard normal.

    Returns (z statistic, two-sided p-value).
    """
    sr_a, sr_b = _sharpe(a), _sharpe(b)
    rho = np.corrcoef(a, b)[0, 1]
    v = 2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)
    if v <= 0:  # identical series: no difference to test
        return 0.0, 1.0
    z = (sr_a - sr_b) / np.sqrt(v / len(a))
    return float(z), float(2 * norm.sf(abs(z)))


def block_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int = config.BOOTSTRAP_SAMPLES,
    block_length: int = config.BOOTSTRAP_BLOCK_LENGTH,
    seed: int = config.RANDOM_SEED,
    trading_days: int = config.TRADING_DAYS,
) -> tuple[float, float, float]:
    """
    Circular block bootstrap of the ANNUALISED Sharpe difference (a minus b).

    Each resample stitches together random blocks of `block_length`
    consecutive days (wrapping around the end, hence "circular") until it is
    as long as the original sample. The same days are used for both
    strategies, so their correlation is preserved.

    p-value: how often a resampled difference, re-centred on the observed
    one, is at least as far from it as the observed difference is from zero.
    This mimics the spread of differences we would see if H0 were true.

    Returns (95% CI lower, 95% CI upper, two-sided p-value).
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    scale = np.sqrt(trading_days)

    observed = (_sharpe(a) - _sharpe(b)) * scale
    diffs = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets) % n).ravel()[:n]
        diffs[i] = (_sharpe(a[idx]) - _sharpe(b[idx])) * scale

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    p_value = float(np.mean(np.abs(diffs - observed) >= abs(observed)))
    return float(lower), float(upper), p_value


def sharpe_significance(
    results: dict[str, BacktestResult],
    risk_free: pd.Series,
    baseline: str = "Equal Weight",
) -> pd.DataFrame:
    """
    Test every strategy's Sharpe ratio against the baseline (equal weight).

    Returns one row per strategy with the annualised Sharpe difference, the
    Memmel z and p-value, and the bootstrap 95% interval and p-value.
    A p-value above 0.05 means the difference is not statistically
    distinguishable from zero at the 5% level.
    """
    base = _excess(results[baseline].returns, risk_free)
    rows = {}
    for name, res in results.items():
        if name == baseline:
            continue
        x = _excess(res.returns, risk_free)
        z, p_memmel = memmel_test(x, base)
        lower, upper, p_boot = block_bootstrap_test(x, base)
        rows[name] = {
            "Sharpe Diff vs 1/N": (_sharpe(x) - _sharpe(base)) * np.sqrt(config.TRADING_DAYS),
            "Memmel z": z,
            "Memmel p": p_memmel,
            "Bootstrap 95% CI Low": lower,
            "Bootstrap 95% CI High": upper,
            "Bootstrap p": p_boot,
        }
    return pd.DataFrame(rows).T
