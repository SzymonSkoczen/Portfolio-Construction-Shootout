"""
Central configuration for the Portfolio Construction Shootout.

Every tunable parameter lives here so that the rest of the code never
hard-codes a number. The robustness checks simply re-run the backtest with
different values of ESTIMATION_WINDOW and TRANSACTION_COST_BPS.
"""

from pathlib import Path

# --- Investable universe ----------------------------------------------------
# Nine SPDR sector ETFs with history back to 1998. XLRE (2015) and XLC (2018)
# are excluded because including them would cut the sample short.
TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

# External benchmark only: reported for comparison, never given to an optimiser.
BENCHMARK = "SPY"

# --- Data -------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = None                 # None = latest available date
RISK_FREE_SERIES = "DTB3"       # FRED 3-month T-bill, annualised % (discount basis)

# --- Backtest ---------------------------------------------------------------
ESTIMATION_WINDOW = 756         # trading days used to estimate mean/covariance (~3 years)
REBALANCE_FREQ = "M"            # month end (rebalance on the last trading day of each month)
TRANSACTION_COST_BPS = 10       # cost per unit of turnover, in basis points
TRADING_DAYS = 252              # used to annualise returns, volatility and Sharpe

# --- Robustness checks ------------------------------------------------------
ROBUSTNESS_WINDOWS = [252, 756, 1260]   # 1, 3 and 5 years
ROBUSTNESS_COSTS_BPS = [0, 10, 25]

# --- Output / paths ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DPI = 150
