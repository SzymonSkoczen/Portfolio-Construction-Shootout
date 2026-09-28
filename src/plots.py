"""
All charts, saved as PNG to output/figures/.

Design rules applied throughout:
- Each strategy keeps ONE colour on every chart (colour follows the strategy).
  The palette was checked for colour-blind separation.
- SPY, the external benchmark, is a dashed dark-grey line so it reads as a
  reference rather than as a sixth strategy.
- Thin lines, light gridlines, and a legend on every multi-series chart.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

import config
from src.backtest import BacktestResult
from src.metrics import drawdown_series

logger = logging.getLogger(__name__)

# --- Colours ----------------------------------------------------------------
# Fixed categorical order (validated for colour-blind separation), assigned
# in the order of src.strategies.STRATEGIES.
STRATEGY_COLOURS = {
    "Equal Weight": "#2a78d6",         # blue
    "Min Variance": "#eb6834",         # orange
    "Max Sharpe": "#1baf7a",           # aqua
    "Risk Parity": "#eda100",          # yellow
    "Min Var (Shrinkage)": "#e87ba4",  # magenta
}
BENCHMARK_COLOUR = "#52514e"          # dark grey, dashed

# Nine sectors: the eight palette hues plus a neutral grey for the ninth.
SECTOR_COLOURS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#8c8b86",
]
SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLI": "Industrials", "XLP": "Cons. Staples",
    "XLY": "Cons. Discretionary", "XLU": "Utilities", "XLB": "Materials",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_WIDTH = 1.4

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.solid_capstyle": "round",
})


def _save(fig: plt.Figure, filename: str, out_dir: Path) -> Path:
    """Save a figure as a PNG at the configured dpi and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_cumulative_returns(
    results: dict[str, BacktestResult],
    benchmark: pd.Series,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    Growth of $1 invested at the start of the out-of-sample period, after costs.

    This is the most intuitive picture of performance: the final height is
    total wealth, and the steepness at any point is the return at that time.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dates = next(iter(results.values())).returns.index
    for name, res in results.items():
        wealth = (1 + res.returns).cumprod()
        ax.plot(wealth, color=STRATEGY_COLOURS[name], lw=LINE_WIDTH, label=name)

    spy = (1 + benchmark.reindex(dates)).cumprod()
    ax.plot(spy, color=BENCHMARK_COLOUR, lw=LINE_WIDTH, ls="--", label=f"{benchmark.name} (benchmark)")

    ax.set_title("Growth of $1, after transaction costs")
    ax.set_ylabel("Portfolio value ($)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:.0f}"))
    ax.legend(loc="upper left")
    return _save(fig, "cumulative_returns.png", out_dir)


def plot_drawdowns(
    results: dict[str, BacktestResult],
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    Drawdown: how far each strategy is below its previous peak at each date.

    Shows the depth and length of losses, e.g. the COVID crash (2020) and the
    2022 rate-hike sell-off, which averages like CAGR hide.

    Five daily series on one axis overlap into an unreadable tangle, so each
    strategy gets its own panel (same scales throughout). Equal weight is drawn
    faintly behind every other panel as the reference to beat.
    """
    names = list(results)
    reference = drawdown_series(results[names[0]].returns)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.9 * len(names)), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        dd = drawdown_series(results[name].returns)
        colour = STRATEGY_COLOURS[name]
        if name != names[0]:
            ax.plot(reference, color=AXIS, lw=0.9, label=f"{names[0]} (reference)")
        ax.fill_between(dd.index, dd, 0, color=colour, alpha=0.12, linewidth=0)
        ax.plot(dd, color=colour, lw=1.0, label=name)
        ax.set_title(f"{name}  (max drawdown {dd.min():.1%})", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.axhline(0, color=AXIS, lw=0.8)

    axes[1].legend(handles=axes[1].get_lines()[:1], loc="lower right")  # explain the grey line once
    fig.supylabel("Drawdown from previous peak", color=INK_SECONDARY, fontsize=10)
    fig.suptitle("Drawdowns: loss from previous peak, after costs", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "drawdowns.png", out_dir)


def plot_weights(result: BacktestResult, out_dir: Path = config.FIGURES_DIR) -> Path:
    """
    Stacked area of target weights at each monthly rebalance.

    Stable bands mean a steady, low-turnover strategy; jagged bands mean the
    optimiser keeps changing its mind, which costs money in trading.
    """
    weights = result.weights
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(
        weights.index,
        weights.T.to_numpy(),
        labels=[f"{t} {SECTOR_NAMES.get(t, '')}".strip() for t in weights.columns],
        colors=SECTOR_COLOURS[: weights.shape[1]],
        edgecolor=SURFACE,   # thin surface-coloured gap between bands
        linewidth=0.6,
    )
    ax.set_title(f"{result.name}: portfolio weights at each rebalance")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.set_xlim(weights.index[0], weights.index[-1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.grid(False)
    # Reverse the legend so it reads top-to-bottom in the same order as the bands.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5))

    slug = result.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return _save(fig, f"weights_{slug}.png", out_dir)


def plot_robustness_sharpe(table: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """
    Sharpe ratio by strategy across estimation windows and cost levels.

    `table` needs columns: window, cost_bps, strategy, Sharpe.
    One panel per estimation window, all sharing the same y-axis so they can
    be compared directly. A line falling steeply from left to right means
    that strategy is very sensitive to trading costs (high turnover).
    """
    windows = sorted(table["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(12, 4.2), sharey=True)
    for ax, window in zip(axes, windows):
        subset = table[table["window"] == window]
        for name, colour in STRATEGY_COLOURS.items():
            rows = subset[subset["strategy"] == name].sort_values("cost_bps")
            ax.plot(
                rows["cost_bps"], rows["Sharpe"], color=colour, lw=LINE_WIDTH,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=name,
            )
        years = window / config.TRADING_DAYS
        ax.set_title(f"{window}-day window (~{years:.0f} yr)", fontsize=11)
        ax.set_xlabel("Transaction cost (bps)")
        ax.set_xticks(sorted(table["cost_bps"].unique()))
    axes[0].set_ylabel("Sharpe ratio (after costs)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Robustness: Sharpe ratio across estimation windows and costs", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "robustness_sharpe.png", out_dir)


def plot_all(
    results: dict[str, BacktestResult],
    benchmark: pd.Series,
    out_dir: Path = config.FIGURES_DIR,
) -> None:
    """Create every base-case chart (the robustness chart is made separately)."""
    plot_cumulative_returns(results, benchmark, out_dir)
    plot_drawdowns(results, out_dir)
    for res in results.values():
        plot_weights(res, out_dir)
