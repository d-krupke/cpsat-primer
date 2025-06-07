import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_performance_profile(
    data: pd.DataFrame,
    instance_column: str,
    strategy_column: str,
    metric_column: str,
    direction: str,
    comparison: str = "relative",
    title: str | None = None,
    highlight_best: bool = False,
    ax: Axes | None = None,
    scale: str | None = None,
    log_base: int = 2,
    figsize: tuple = (9, 6),
) -> Axes:
    """
    Plot a performance profile, either on a relative-ratio basis or absolute-difference basis:
      - For comparison="relative":
          x-axis: performance ratio τ (log scale if τ_max > 10)
          τ = (value / best) if direction="min", or τ = (best / value) if direction="max".
      - For comparison="absolute":
          x-axis: absolute difference Δ = (value - best) if direction="min",
                                      or Δ = (best - value) if direction="max".
      - y-axis: proportion of problems with τ (or Δ) ≤ x for each solver.
      - If highlight_best=True, detect and bold the dominating solver curve (AUC in appropriate space).
      - Ensures a reasonable number of ticks on the x-axis.

    Args:
        data: DataFrame with columns [instance, strategy, metric].
        instance_column: column name identifying each problem instance.
        strategy_column: column name identifying each solver/strategy.
        metric_column: column name of the performance metric (e.g. runtime or cost).
        direction: "min" if lower metric → better, "max" if higher → better.
        comparison: "relative" or "absolute".
        title: Optional plot title.
        highlight_best: If True, find the solver with largest AUC and draw it in bold.
        ax: An existing matplotlib Axes to draw into. If None, a new Figure+Axes will be created using figsize.
        scale: x-axis scale override ("linear" or "log"); if None, chosen automatically.
        log_base: base for log scale if used (default 2).
        figsize: Tuple (width, height). Only used if ax is None.

    Returns:
        The matplotlib Axes containing the performance profile.
    """
    if direction not in ("min", "max"):
        raise ValueError("`direction` must be 'min' or 'max'.")
    if comparison not in ("relative", "absolute"):
        raise ValueError("`comparison` must be 'relative' or 'absolute'.")

    # 1) Compute best value per instance
    best_val = data.groupby(instance_column)[metric_column].agg(direction)

    # 2) Pivot to get per-instance × per-strategy medians
    pivot = (
        data.groupby([instance_column, strategy_column])[metric_column]
        .median()
        .unstack(fill_value=np.nan)
    )

    # 3) Build comparison matrix C[p, s]
    comp = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)

    if comparison == "relative":
        for strat in pivot.columns:
            if direction == "min":
                comp[strat] = pivot[strat] / best_val
            else:  # direction == "max"
                comp[strat] = best_val / pivot[strat]
        comp = comp.replace([np.inf, -np.inf, 0.0], np.nan)

    else:  # comparison == "absolute"
        for strat in pivot.columns:
            if direction == "min":
                comp[strat] = pivot[strat] - best_val
            else:  # direction == "max"
                comp[strat] = best_val - pivot[strat]
        comp = comp.replace([np.inf, -np.inf], np.nan)

    # 4) Collect all distinct x-values (τ or Δ), including baseline
    all_vals = comp.values.flatten()
    finite_vals = all_vals[np.isfinite(all_vals)]
    baseline = 1.0 if comparison == "relative" else 0.0
    all_x = np.unique(np.sort(finite_vals))
    all_x = np.concatenate(([baseline], all_x))
    all_x = np.unique(np.sort(all_x))

    # 5) Build performance-profile DataFrame ρ(x)
    n_instances = comp.shape[0]
    profile = pd.DataFrame(index=all_x, columns=comp.columns, dtype=float)

    for x in all_x:
        leq = (comp <= x).sum(axis=0)
        profile.loc[x] = leq / n_instances

    # 6) Identify dominating solver if requested (max AUC)
    best_solver = None
    if highlight_best:
        if comparison == "relative":
            # integrate ρ(τ) w.r.t. log(τ)
            log_x = np.log(all_x)
            areas = {}
            for strat in profile.columns:
                y = profile[strat].astype(float).values
                areas[strat] = np.trapz(y, x=log_x)
            best_solver = max(areas, key=areas.get)
        else:
            # integrate ρ(Δ) w.r.t. Δ
            areas = {}
            for strat in profile.columns:
                y = profile[strat].astype(float).values
                areas[strat] = np.trapz(y, x=all_x)
            best_solver = max(areas, key=areas.get)

    # 7) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 8) Determine scale if not overridden
    if scale is None:
        if comparison == "relative" and all_x[-1] > 10:
            use_log = True
        else:
            use_log = False
    else:
        use_log = scale == "log"

    # 9) Plot each solver’s curve
    for strat in profile.columns:
        y = profile[strat].astype(float)
        if highlight_best and strat == best_solver:
            ax.step(all_x, y, where="post", label=strat, linewidth=3.0, alpha=1.0)
        else:
            ax.step(
                all_x,
                y,
                where="post",
                label=strat,
                linewidth=1.5,
                alpha=0.6 if highlight_best else 1.0,
            )

    # 10) Axis scaling and limits
    if comparison == "relative":
        if use_log:
            ax.set_xscale("log", base=log_base)
            ax.set_xlim(all_x[1], all_x[-1] * 1.1)
        else:
            ax.set_xscale("linear")
            ax.set_xlim(1.0, all_x[-1] * 1.1)
        xlabel = (
            f"Within this factor of the best (log{log_base} scale)"
            if use_log
            else "Within this factor of the best (linear scale)"
        )
    else:  # absolute
        ax.set_xscale("linear")
        ax.set_xlim(0.0, all_x[-1] * 1.1)
        xlabel = "Absolute difference from the best"

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Proportion of problems", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=14)
    else:
        ax.set_title("Performance Profile", fontsize=14, pad=14)

    ax.axvline(x=baseline, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # 11) Legend inside lower right
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    return ax
