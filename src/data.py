"""
Data pipeline: download, cache and clean prices; compute daily returns and
the daily risk-free rate.

Sources
-------
- Prices: Yahoo Finance via `yfinance`, using *adjusted* closes
  (`auto_adjust=True`). Adjusted prices fold dividends and splits back into
  the price series, so a simple percentage change is a total return.
- Risk-free rate: FRED series DTB3, the 3-month US Treasury bill rate.

Every download is cached as CSV in `data/raw/`, so later runs are fast,
work offline, and do not hit Yahoo's rate limits.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

PRICES_CACHE = "prices.csv"
RISK_FREE_CACHE = "risk_free.csv"


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: Path = config.RAW_DATA_DIR,
    refresh: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Load daily adjusted close prices, from cache if possible, else from Yahoo.

    Adjusted closes are used because they include reinvested dividends. Sector
    ETFs such as XLU and XLP pay meaningful dividends, so raw closes would
    understate their returns and bias the comparison between strategies.

    Parameters
    ----------
    tickers : list of Yahoo tickers to download (assets and benchmark).
    start, end : date range; `end=None` means "up to the latest date".
    cache_dir : folder holding the cached CSV.
    refresh : if True, ignore the cache and download again.
    max_retries : Yahoo sometimes rate-limits; retry with a growing pause.

    Returns
    -------
    DataFrame of prices, indexed by date, one column per ticker.
    """
    cache_path = Path(cache_dir) / PRICES_CACHE

    # Use the cache only if it holds every ticker we need; otherwise a change
    # to config.TICKERS would silently reuse stale data.
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            logger.info("Loaded prices from cache: %s", cache_path)
            return cached[tickers]
        logger.info("Cache is missing some tickers; downloading again.")

    prices = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading prices for %d tickers (attempt %d)...", len(tickers), attempt)
            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True, progress=False
            )
            prices = raw["Close"]
            # A failed ticker comes back as an all-NaN column rather than an error.
            if prices.empty or prices.isna().all().any():
                raise RuntimeError("some tickers returned no data")
            break
        except Exception as exc:  # network errors, rate limits, empty data
            logger.warning("Price download failed: %s", exc)
            if attempt == max_retries:
                raise RuntimeError("Could not download prices from Yahoo Finance.") from exc
            time.sleep(5 * attempt)

    prices = prices[tickers]  # keep the configured column order
    prices.index.name = "Date"

    # If we download during US trading hours, Yahoo returns today's row with a
    # live intraday price, not a closing price. Keep completed sessions only,
    # so results do not change depending on the time of day the code is run.
    today_new_york = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    if prices.index[-1] >= today_new_york:
        logger.info("Dropping today's incomplete trading session (%s).", prices.index[-1].date())
        prices = prices[prices.index < today_new_york]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    logger.info("Saved %d rows of prices to %s", len(prices), cache_path)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to simple daily returns: r_t = P_t / P_{t-1} - 1.

    Simple (not log) returns are used because they aggregate across assets:
    a portfolio's return is the weighted sum of its assets' simple returns.

    Dates where any ticker has a missing price are dropped first, so every
    remaining date is a valid trading day for the whole universe. The number
    of dropped dates is logged so data gaps never go unnoticed.
    """
    missing = prices.isna().any(axis=1)
    if missing.any():
        logger.info("Dropping %d dates with missing prices.", int(missing.sum()))
    clean = prices.loc[~missing]

    # fill_method=None: never fill a gap with a stale price (that would fake a
    # 0% return); the gaps were already removed above.
    returns = clean.pct_change(fill_method=None).iloc[1:]
    logger.info(
        "Computed returns: %d days from %s to %s.",
        len(returns), returns.index[0].date(), returns.index[-1].date(),
    )
    return returns


def load_risk_free(
    dates: pd.DatetimeIndex,
    start: str,
    cache_dir: Path = config.RAW_DATA_DIR,
    refresh: bool = False,
    trading_days: int = config.TRADING_DAYS,
) -> pd.Series:
    """
    Daily risk-free rate aligned to `dates`, from the 3-month T-bill (FRED DTB3).

    DTB3 is quoted as an annualised percentage (e.g. 4.5 means 4.5% a year).
    We convert it to a daily rate by compounding:
        r_daily = (1 + r_annual)^(1 / trading_days) - 1
    This daily rate is what the Sharpe ratio subtracts from daily returns.

    FRED has no values on US bank holidays, so the rate is forward-filled:
    the latest known T-bill yield is the best estimate for that day.

    If FRED cannot be reached and there is no cache, the rate is set to 0 with
    a warning. The run can still finish, but Sharpe ratios then measure raw
    return per unit of risk rather than excess return.
    """
    cache_path = Path(cache_dir) / RISK_FREE_CACHE
    annual_pct = None

    if cache_path.exists() and not refresh:
        annual_pct = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        logger.info("Loaded risk-free rate from cache: %s", cache_path)
    else:
        try:
            import pandas_datareader.data as web

            logger.info("Downloading %s from FRED...", config.RISK_FREE_SERIES)
            annual_pct = web.DataReader(config.RISK_FREE_SERIES, "fred", start=start).iloc[:, 0]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            annual_pct.to_csv(cache_path)
        except Exception as exc:
            logger.warning("FRED download failed (%s). Using a risk-free rate of 0.", exc)

    if annual_pct is None:
        return pd.Series(0.0, index=dates, name="rf")

    daily = (1 + annual_pct / 100) ** (1 / trading_days) - 1
    # Align to trading dates: forward-fill holidays, back-fill the first day if
    # FRED's first observation is a holiday (e.g. 1 January).
    daily = daily.reindex(daily.index.union(dates)).ffill().bfill().reindex(dates)
    daily.name = "rf"
    return daily


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the full data pipeline with the settings in `config.py`.

    Returns
    -------
    asset_returns : daily returns of the investable universe (config.TICKERS)
    benchmark_returns : daily returns of the benchmark (SPY), same dates
    risk_free : daily risk-free rate, same dates
    """
    tickers = config.TICKERS + [config.BENCHMARK]
    prices = download_prices(tickers, config.START_DATE, config.END_DATE, refresh=refresh)
    returns = compute_returns(prices)
    risk_free = load_risk_free(returns.index, config.START_DATE, refresh=refresh)
    return returns[config.TICKERS], returns[config.BENCHMARK], risk_free
