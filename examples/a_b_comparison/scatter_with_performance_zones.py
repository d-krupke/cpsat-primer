"""
This module contains functions to plot a scatter comparison of baseline and new values with performance areas highlighted.

You can freely use and distribute this code under the MIT license.

Changelog:
    2024-08-27: First version
    2024-08-29: Added lines to the diagonal to help with reading the plot

(c) 2024 Dominik Krupke, https://github.com/d-krupke/cpsat-primer
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_performance_scatter(
    ax,
    baseline: pd.Series,
    new_values: pd.Series,
    lower_is_better: bool = True,
    title: str = "",
    **kwargs,
):
    """
    Plot a scatter comparison of baseline and new values with performance areas highlighted.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        baseline (pd.Series): Series of baseline values.
        new_values (pd.Series): Series of new values.
        lower_is_better (bool): If True, lower values indicate better performance.
        title (str): Title of the plot.
        **kwargs: Additional keyword arguments for customization (e.g., 'color', 'marker').
    """
    if not isinstance(baseline, pd.Series) or not isinstance(new_values, pd.Series):
        raise ValueError("Both baseline and new_values should be pandas Series.")
    if baseline.size != new_values.size:
        raise ValueError("Both Series should have the same length.")

    scatter_kwargs = {
        "color": kwargs.get("color", "blue"),
        "marker": kwargs.get("marker", "x"),
        "label": kwargs.get("label", "Data Points"),
    }

    line_kwargs = {
        "color": kwargs.get("line_color", "k"),
        "linestyle": kwargs.get("line_style", "--"),
        "label": kwargs.get("line_label", "No Change"),
    }

    fill_improve_kwargs = {
        "color": kwargs.get("improve_color", "green"),
        "alpha": kwargs.get("improve_alpha", 0.3),
        "label": kwargs.get("improve_label", "Improved Performance"),
    }

    fill_decline_kwargs = {
        "color": kwargs.get("decline_color", "red"),
        "alpha": kwargs.get("decline_alpha", 0.3),
        "label": kwargs.get("decline_label", "Declined Performance"),
    }

    # Replace inf values with NaN
    baseline = baseline.replace([np.inf, -np.inf], np.nan)
    new_values = new_values.replace([np.inf, -np.inf], np.nan)

    max_val = max(baseline.max(skipna=True), new_values.max(skipna=True)) * 1.05
    min_val = min(baseline.min(skipna=True), new_values.min(skipna=True)) * 0.95

    # get indices of NA values
    na_indices = baseline.isna() | new_values.isna()

    if lower_is_better:
        # replace NA values with max_val
        baseline = baseline.fillna(max_val)
        new_values = new_values.fillna(max_val)
    else:
        # replace NA values with min_val
        baseline = baseline.fillna(min_val)
        new_values = new_values.fillna(min_val)

    # plot the na_indices with a different marker
    if na_indices.any():
        ax.scatter(
            baseline[na_indices],
            new_values[na_indices],
            marker="s",
            color=scatter_kwargs["color"],
            label="NA Values",
            zorder=2,
        )

    # add the rest of the data points
    ax.scatter(
        baseline[~na_indices], new_values[~na_indices], **scatter_kwargs, zorder=2
    )

    ax.plot([min_val, max_val], [min_val, max_val], zorder=1, **line_kwargs)

    x = np.linspace(min_val, max_val, 500)
    if lower_is_better:
        ax.fill_between(x, min_val, x, zorder=0, **fill_improve_kwargs)
        ax.fill_between(x, x, max_val, zorder=0, **fill_decline_kwargs)
    else:
        ax.fill_between(x, x, max_val, zorder=0, **fill_improve_kwargs)
        ax.fill_between(x, min_val, x, zorder=0, **fill_decline_kwargs)

    # draw thin lines to the diagonal to help with reading the plot.
    # A problem without lines is that one tends to use the distance
    # to the diagonal as a measure of performance, which is not correct.
    # Instead, it is `y-x` that should be used.
    for old_val, new_val in zip(baseline[~na_indices], new_values[~na_indices]):
        if lower_is_better and new_val < old_val:
            ax.plot(
                [old_val, old_val],
                [old_val, new_val],
                color="green",
                linewidth=1.0,
                zorder=1,
            )
        elif not lower_is_better and new_val > old_val:
            ax.plot(
                [old_val, old_val],
                [old_val, new_val],
                color="green",
                linewidth=1.0,
                zorder=1,
            )
        elif lower_is_better and new_val > old_val:
            ax.plot(
                [old_val, old_val],
                [old_val, new_val],
                color="red",
                linewidth=1.0,
                zorder=1,
            )
        elif not lower_is_better and new_val < old_val:
            ax.plot(
                [old_val, old_val],
                [old_val, new_val],
                color="red",
                linewidth=1.0,
                zorder=1,
            )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(kwargs.get("xlabel", "Baseline"))
    ax.set_ylabel(kwargs.get("ylabel", "New Values"))
    if title:
        ax.set_title(title)
    ax.legend()


def plot_comparison_grid(
    baseline_data: pd.DataFrame,
    new_data: pd.DataFrame,
    metrics: list[tuple[str, str]],
    n_cols: int = 4,
    figsize: tuple[int, int] | None = None,
    **kwargs,
):
    """
    Plot a grid of performance comparisons for multiple metrics.

    Parameters:
        baseline_data (pd.DataFrame): DataFrame containing the baseline data.
        new_data (pd.DataFrame): DataFrame containing the new data.
        metrics (list of tuple of str): List of tuples containing column names and comparison direction ('min' or 'max').
        n_cols (int): Number of columns in the grid.
        figsize (tuple of int): Figure size (width, height).
        **kwargs: Additional keyword arguments to pass to individual plot functions.

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plots.
        axes (np.ndarray): Array of axes objects corresponding to the subplots.
    """
    n_metrics = len(metrics)
    n_cols = min(n_cols, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    # Validate columns and directions
    for column, direction in metrics:
        if direction not in {"min", "max"}:
            raise ValueError("The direction should be either 'min' or 'max'.")
        if column not in baseline_data.columns or column not in new_data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

    # Validate index alignment
    if not baseline_data.index.equals(new_data.index):
        raise ValueError("Indices of the DataFrames do not match.")

    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (column_name, direction) in zip(axes, metrics):
        plot_performance_scatter(
            ax,
            baseline_data[column_name],
            new_data[column_name],
            lower_is_better=(direction == "min"),
            title=column_name,
            **kwargs,
        )

    # Turn off any unused subplots
    for ax in axes[n_metrics:]:
        ax.axis("off")

    plt.tight_layout()

    return fig, axes
