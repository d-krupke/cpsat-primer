<!-- EDIT THIS PART VIA 08_benchmarking.md -->

<a name="08-benchmarking"></a>

## Benchmarking your Model

<!-- START_SKIP_FOR_README -->

![Cover Image Benchmarking](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_5.webp)

<!-- STOP_SKIP_FOR_README -->

This chapter explores methods for comparing the performance of different models
applied to complex problems, where the basic model leaves room for improvement -
either in runtime or in solution quality, especially when the model cannot
always be solved to optimality. As a scientist writing a paper on a new model
(or, more likely, a new algorithm that internally uses a model), this will be
the default case as research on a problem that can already be solved well is
hard to publish. Whether you aim merely to evaluate whether your new approach
outperforms the current one or intend to prepare a formal scientific
publication, you face the same challenges; in the latter case, however, the
process becomes more extensive and formalized (but this may also be true for the
first case depending on your manager).

During the explorative phase, when you probe different ideas, you will likely
select one to five instances that you can run quickly and compare. However, for
most applications, this number is insufficient, and you risk overfitting your
model to these instances - gaining performance improvements on them but
sacrificing performance on others. You may even limit your model’s ability to
solve certain instances.

A classic example involves deactivating specific CP-SAT search strategies or
preprocessing steps that have not yielded benefits on the selected instances. If
your instance set is large enough, the risk is low; however, if you have only a
few instances, you may remove the single strategy necessary to solve a
particular class of problems. Modern solvers include features that impose a
modest overhead on simple instances but enable solving otherwise intractable
cases. This trade-off is worthwhile: do not sacrifice the ability to solve
complex instances for a marginal performance gain on simple ones. Therefore,
always benchmark your changes properly before deploying them to production, even
if you do not plan to publish your results scientifically.

The no‐free‐lunch theorem and timeouts complicate benchmarking more than you
might have anticipated. The no‐free‐lunch theorem asserts that no single
algorithm outperforms all others across every instance, which is especially true
for NP‐hard problems. Consequently, improving performance on some instances
often coincides with degradations on others. It is essential to assess whether
the gains justify the losses.

Another challenge arises when imposing a time limit to prevent any individual
instance from running indefinitely; without it, benchmarks can take
prohibitively long. Yet comparing aborted runs to those that complete within the
time limit poses a dilemma: disqualifying models that time out may leave no
viable candidate, since with sufficient instances, any solver will be unlucky at
least once. Thus, simple exclusion is not an option for most applications.
Timeouts introduce "unknowns" into your results: a solver might have succeeded
given just one more millisecond, or it might have been trapped in an endless
loop. This uncertainty complicates the computation of accurate statistics.

Let us examine the performance of CP-SAT, Gurobi, and Hexaly on a Nurse
Rostering Problem. Nurse rostering is a complex yet common problem in which
nurses must be assigned to shifts while satisfying a variety of constraints.
Since CP-SAT, Gurobi, and Hexaly differ significantly in their underlying
algorithms, the comparison reveals pronounced performance differences. However,
such patterns can also be observed when using the same solver across instances,
albeit usually not as pronounced.

The following two plots illustrate the value of the incumbent solution (i.e.,
the best solution found so far) and the best proven lower bound during the
search. These are challenging instances, and only Gurobi is able to find an
optimal solution.

Notably, the best-performing solver changes depending on the allotted
computation time. For this problem, Hexaly excels at finding good initial
solutions quickly but tends to stall thereafter. CP-SAT requires slightly more
time to get started but demonstrates steady progress. In contrast, Gurobi begins
slowly but eventually makes substantial improvements.

So, which solver is best for this problem?

|                                                                                              ![NRP Instance 19](https://github.com/d-krupke/cpsat-primer/blob/main/images/nrp_19.png?raw=true)                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Performance comparison of CP-SAT, Gurobi, and Hexaly on instance 19 of the Nurse Rostering Problem Benchmark. Hexaly starts strong but is eventually overtaken by CP-SAT. Gurobi surpasses Hexaly near the end by a small margin. CP-SAT and Gurobi converge to nearly the same lower bound. |

|                                                                                                                                                                               ![NRP Instance 20](https://github.com/d-krupke/cpsat-primer/blob/main/images/nrp_20.png?raw=true)                                                                                                                                                                               |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Performance comparison of CP-SAT, Gurobi, and Hexaly on instance 20 of the Nurse Rostering Problem Benchmark. Hexaly again performs well early but is outperformed by CP-SAT. Gurobi maintains a poor incumbent value for most of the runtime but eventually makes a significant improvement and proves optimality. The optimal solution is visibly superior to CP-SAT's best solution. CP-SAT is unable to prove a meaningful lower bound for this instance. |

> [!WARNING]
>
> These two plots - and even this specific problem - are insufficient to draw
> definitive conclusions about the overall performance of the solvers.
> Nevertheless, it is remarkable that our beloved open-source solver, CP-SAT,
> performs so well against the commercial solvers Gurobi and Hexaly in this
> context.

If provably optimal solutions are required, Gurobi may be the most suitable
choice; however, the likelihood of achieving optimality is low for most
instances. If your instances are expected to grow in size and require fast
solutions, Hexaly might be preferable, as it appears highly effective at finding
good solutions quickly. This observation is also supported by preliminary
results on larger instances, for which neither CP-SAT nor Gurobi find any
feasible solution. CP-SAT, by contrast, offers a strong compromise between the
two: although it starts more slowly, it maintains consistent progress throughout
the search.

The first step is to determine your specific requirements and how best to
measure solver performance accordingly. It is not feasible to manually plot
performance for every instance and assign scores based on subjective
impressions; such an approach does not scale and lacks objectivity and
reproducibility. Instead, you should define a concrete metric that accurately
reflects your goals. One strategy is to carefully select benchmark instances
that are still likely to be solved to optimality, with the expectation that
performance trends will generalize to larger instances. While no evaluation
method will be perfect, it is essential to remain aware of potential threats to
the validity of your results. Let us go through some common scenarios.

## Common Benchmarking Scenarios

Several common benchmarking scenarios arise in practice. To choose the
appropriate visualization or evaluation method, it is useful to take a moment to
identify which scenario applies to your case—and to recognize which tools are
best suited to other situations. Do not simply go for the prettiest and most
complex plot; instead, select the one that best fits your needs.

1. **Instances are always solved to optimality, and only runtime matters.** This
   is the simplest benchmarking scenario. If every instance can be solved to
   optimality and your only concern is runtime, you can summarize performance
   using the mean runtime or visualize it using a basic box plot. The primary
   challenge here lies in choosing the appropriate type of mean (e.g.,
   arithmetic, geometric, harmonic) and selecting representative instances that
   reflect production-like conditions. For this case, the rest of the chapter
   may be skipped.

2. **Optimal or feasible solutions are sought, but may not always be found
   within the time limit.** When timeouts occur, runtimes for unsolved instances
   become unknown, making traditional means unreliable. In such cases, **cactus
   plots** are an effective way to visualize solver performance, even in the
   presence of incomplete data.

3. **The goal is to find the best possible solution within a fixed time limit.**
   Here, the focus is on **solution quality under time constraints**, rather
   than on whether optimality is reached. **Performance plots** are especially
   suitable for this purpose, as they reveal how closely each solver or model
   approaches the best-known solution across the benchmark set.

4. **Scalability analysis: how performance evolves with instance size.** If you
   are analyzing how well a model scales—i.e., how large an instance it can
   solve to optimality and how the optimality gap grows thereafter—**split
   plots** are a good choice. They show runtime for solved instances and
   optimality gap for those that exceed the time limit, allowing for a unified
   view of scalability.

5. **Multi-metric performance comparison against a baseline.** When you want a
   quick, intuitive overview of how your model performs across several
   metrics—such as runtime, objective value, and lower bound—**scatter plots
   with performance zones** are ideal. They provide a clear comparative
   visualization, making it easy to spot outliers and trade-offs across
   dimensions.

### Quickly Comparing to a Baseline Using Scatter Plots

Scatter plots with performance zones are, in my experience, highly effective for
quickly comparing the performance of a prototype against a baseline across
multiple metrics. While these plots do not provide a formal quantitative
evaluation, they offer a clear visual overview of how performance has shifted.
Their key advantages are their intuitive readability and their ability to
accommodate `NaN` values. They are particularly useful for identifying outliers,
though they can be less effective when too many points overlap or when data
ranges vary significantly.

Consider the following example table, which compares a basic optimization model
with a prototype model in terms of runtime, objective value, and lower bound.
The runtime is capped at 90 seconds, and if no optimal solution is found within
this limit, the objective value is set to `NaN`. A run only terminates before
the time limit if an optimal solution is found.

<details><summary>Example Data</summary>

| instance_name | strategy  | runtime |   objective | lower_bound |
| :------------ | :-------- | ------: | ----------: | ----------: |
| att48         | Prototype | 89.8327 |       33522 |       33522 |
| att48         | Baseline  | 90.1308 |       33522 |       33369 |
| eil101        | Prototype | 90.0948 |         629 |         629 |
| eil101        | Baseline  | 43.8567 |         629 |         629 |
| eil51         | Prototype | 84.8225 |         426 |         426 |
| eil51         | Baseline  | 3.05334 |         426 |         426 |
| eil76         | Prototype | 90.2696 |         538 |         538 |
| eil76         | Baseline  | 4.09839 |         538 |         538 |
| gil262        | Prototype | 90.3314 |       13817 |        2368 |
| gil262        | Baseline  | 90.8782 |        3141 |        2240 |
| kroA100       | Prototype | 90.5127 |       21282 |       21282 |
| kroA100       | Baseline  | 90.0241 |       22037 |       20269 |
| kroA150       | Prototype |  90.531 |       27249 |       26420 |
| kroA150       | Baseline  | 90.3025 |       27777 |       24958 |
| kroA200       | Prototype | 90.0019 |      176678 |       29205 |
| kroA200       | Baseline  | 90.7658 |       32749 |       27467 |
| kroB100       | Prototype | 90.1334 |       22141 |       22141 |
| kroB100       | Baseline  | 90.5845 |       22729 |       21520 |
| kroB150       | Prototype | 90.7107 |      128751 |       26016 |
| kroB150       | Baseline  | 90.9659 |       26891 |       25142 |
| kroB200       | Prototype | 90.7931 |      183078 |       29334 |
| kroB200       | Baseline  | 90.3594 |       34481 |       27708 |
| kroC100       | Prototype | 90.5131 |       20749 |       20749 |
| kroC100       | Baseline  | 90.3035 |       21118 |       20125 |
| kroD100       | Prototype | 90.0728 |       21294 |       21294 |
| kroD100       | Baseline  | 90.2563 |       21294 |       20267 |
| kroE100       | Prototype | 90.4515 |       22068 |       22053 |
| kroE100       | Baseline  | 90.6112 |       22341 |       21626 |
| lin105        | Prototype | 90.4714 |       14379 |       14379 |
| lin105        | Baseline  | 90.6532 |       14379 |       13955 |
| lin318        | Prototype | 90.8489 |      282458 |       41384 |
| lin318        | Baseline  | 90.5955 |      103190 |       39016 |
| linhp318      | Prototype | 90.9566 |         nan |       41412 |
| linhp318      | Baseline  | 90.7038 |       84918 |       39016 |
| pr107         | Prototype | 90.3708 |       44303 |       44303 |
| pr107         | Baseline  | 90.4465 |       45114 |       27784 |
| pr124         | Prototype | 90.1689 |       59167 |       58879 |
| pr124         | Baseline  | 90.8673 |       60760 |       52392 |
| pr136         | Prototype | 90.0296 |       96781 |       96772 |
| pr136         | Baseline  | 90.2636 |       98850 |       89369 |
| pr144         | Prototype | 90.3141 |       58537 |       58492 |
| pr144         | Baseline  | 90.6465 |       59167 |       33809 |
| pr152         | Prototype | 90.4742 |       73682 |       73682 |
| pr152         | Baseline  | 90.8629 |       79325 |       46604 |
| pr226         | Prototype | 90.1845 | 1.19724e+06 |       74474 |
| pr226         | Baseline  | 90.6676 |      103271 |       55998 |
| pr264         | Prototype | 90.4012 |      736226 |       41020 |
| pr264         | Baseline  | 90.9642 |       68802 |       37175 |
| pr299         | Prototype | 90.3325 |         nan |       47375 |
| pr299         | Baseline  |   90.19 |      120489 |       45594 |
| pr439         | Prototype | 90.5761 |         nan |       95411 |
| pr439         | Baseline  |  90.459 |      834126 |       93868 |
| pr76          | Prototype | 90.2718 |      108159 |      107727 |
| pr76          | Baseline  | 90.2951 |      110331 |      105340 |
| st70          | Prototype | 90.2824 |         675 |         675 |
| st70          | Baseline  | 90.1484 |         675 |         663 |

</details>

From the table, we can already spot some fundamental issues — for example, the
prototype fails on three instances, and several instances yield significantly
worse results than the baseline. However, when we turn to the scatter plots with
performance zones, such anomalies become immediately apparent.

|                                                                                                                                              ![Scatter Plot with Performance Zones](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scatter_tsp.png)                                                                                                                                              |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Scatter plot comparing the performance of a prototype model against a baseline model across three metrics: runtime, objective value, and lower bound. The x-axis represents the baseline model's performance; the y-axis shows the prototype model's performance. Color-coded zones indicate relative performance levels, making it easier to identify where the prototype outperforms or underperforms the baseline. |

For runtime, both models typically hit the time limit, so there is limited
variation to observe. However, the baseline model solves a few instances
significantly faster, whereas the prototype consistently uses the full time
limit. For the objective value, both models produce similar results on most
instances. Yet, particularly on the larger instances, the prototype yields
either very poor or no solutions at all.

Interestingly, the lower bounds produced by the prototype are much better for
some instances. This improvement was not obvious from a cursory review of the
table but becomes immediately noticeable in the plots.

Scatter plots are also highly effective when working with multiple performance
metrics, particularly when you want to ensure that gains in one metric do not
come at the expense of unacceptable losses in another. In practice, it is often
difficult to precisely quantify the relative importance of each metric from the
outset. The intuitive nature of these plots offers a valuable overview, serving
as a visual aid before you commit to a specific performance metric. The example
below illustrates a hypothetical scenario involving a vehicle routing problem.

|                                                                                                                                                             ![Scatter Plot for Multi-Objectives](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scatter_performance_zones.png)                                                                                                                                                              |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Scatter plots illustrating performance trade-offs across multiple metrics in a hypothetical vehicle routing problem. These plots help assess whether improvements in one metric come at the cost of significant regressions in another. Their intuitive layout makes them especially useful when metric priorities are not yet clearly defined, offering a quick overview of relative performance and highlighting outliers across different algorithm versions. |

<details>
<summary>Here is the code I used to generate the plots. You can freely copy and use it.</summary>

```python
# MIT License
# Dominik Krupke, 2025
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
            label="N/A Values",
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
    for old_val, new_val in zip(baseline, new_values):
        if pd.isna(old_val) and pd.isna(new_val):
            continue
        if pd.isna(old_val):
            old_val = min_val if lower_is_better else max_val
        if pd.isna(new_val):
            new_val = min_val if lower_is_better else max_val
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
    suptitle: str = "",
    subplot_kwargs: dict | None = None,
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
        suptitle (str): Title for the entire figure.
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
        # Merge kwargs and subplot_kwargs[column_name] (if present) into a new dict
        merged_kwargs = dict(kwargs)
        if subplot_kwargs and column_name in subplot_kwargs:
            merged_kwargs.update(subplot_kwargs[column_name])
        plot_performance_scatter(
            ax,
            baseline_data[column_name],
            new_data[column_name],
            lower_is_better=(direction == "min"),
            title=column_name,
            **merged_kwargs,
        )

    # Turn off any unused subplots
    for ax in axes[n_metrics:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axes
```

</details>

### Success-Based Benchmarking: Cactus Plots and PAR Metrics

The SAT community frequently uses cactus plots (also known as survival plots) to
effectively compare the time to success of different solvers on a benchmark set,
even when timeouts occur. If you are dealing with a pure constraint satisfaction
problem, this approach is directly applicable. However, it can also be extended
to other binary success indicators — such as proving optimality, even under
optimality tolerances.

Additionally, the **PAR10** metric is commonly used to summarize solver
performance on a benchmark set. It is defined as the average time a solver takes
to solve an instance, where unsolved instances (within the time limit) are
penalized by assigning them a runtime equal to 10 times the cutoff. Variants
like **PAR2**, which use a penalty factor of 2 instead of 10, are also
encountered. While a factor of 10 is conventional, it remains an arbitrary
choice. Ultimately, you must decide how to handle unknowns—instances not solved
within the time limit—since you only know that their actual runtime exceeds the
cutoff. If an explicit performance metric is required to declare a winner,
PAR-style metrics are widely accepted but come with notable limitations.

To gain a more nuanced view of solver performance, **cactus plots** are often
employed. In these plots, each solver is represented by a line where each point
$(x, y)$ indicates that $x$ benchmark instances were solved within $y$ seconds.

| ![Cactus Plot 1](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/cactus_plot.png?raw=true) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                     Each point $(x, y)$ shows that $x$ instances were solved within $y$ seconds.                                      |

The mean PAR10 scores for the four strategies in the example above are as
follows:

| Strategy             |       PAR10 |
| :------------------- | ----------: |
| AddCircuit           |  512.133506 |
| Dantzig (Gurobi)     |   66.452202 |
| Iterative Dantzig    |  752.412118 |
| Miller-Tucker-Zemlin | 1150.014846 |

In case you are wondering, this is some data on solving the Traveling Salesman
Problem (TSP) with different strategies. Gurobi dominates, but it is well-known
that Gurobi is excellent at solving TSP.

If the number of solvers or models under comparison is not too large, you can
also use a variation of the cactus plot to show solver performance under
different **optimality tolerances**. This allows you to examine how much
performance improves when the tolerance is relaxed. However, if your primary
interest lies in **solution quality**, performance plots are likely to be more
appropriate.

In the following example, different optimality tolerances reveal a visible
performance improvement for the strategies `AddCircuit` and
`Miller-Tucker-Zemlin`. For the other two strategies, the impact of tolerance
changes is minimal. This variation of the cactus plot can also be applied to
compare solver performance across different benchmark sets, especially if you
suspect significant variation across instance categories.

| ![Cactus Plot with Optimality Tolerances](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/cactus_plot_opt_tol.png?raw=true) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|             Each line style represents an optimality tolerance. The plot shows how many instances ($y$) can be solved within a given time limit ($x$) for each tolerance.              |

It is also common practice to include a **virtual best** line in the cactus
plot. This line indicates, for each instance, the best time achieved by any
solver. Although it does not represent an actual solver, it serves as a valuable
reference to evaluate the potential for solver complementarity. If one solver
clearly dominates, the virtual best line will coincide with its curve. However,
if the lines diverge, it suggests that no single solver is universally
superior—a case of the “no free lunch” principle. Even if one solver performs
best on 90% of instances, the remaining 10% may be better handled by
alternatives. The greater the gap between the best actual solver and the virtual
best, the stronger the case for a portfolio approach.

### Performance Plots for Solution Quality within a Time Limit

When dealing with instances that typically cannot be solved to optimality,
**performance plots** are often more appropriate than cactus plots. These plots
illustrate the relative performance of different models or solvers on a set of
instances, usually under a fixed time limit. At the leftmost point of the plot
(where $x = 1$), each solver’s line indicates the proportion of instances for
which it achieved the best-known solution (not necessarily exclusively). Then
its $(x,y)$ coordinates indicate the proportion $y$ of instances for which the
solver achieved a solution that is at most $x$ times worse than the best known
solution. For example, if a solver has a point at $(1.05, 0.8)$, it means that
it found a solution within 5% of the best-known solution for 80% of the
instances. The x-axis is usually on a logarithmic scale, allowing for a more
detailed view of the performance differences. Often, a logarithmic scale is used
for the x-axis, especially when the performance ratios vary widely. However,
down below we use a linear scale because the values are close to 1.

In the example below, based on the **Capacitated Vehicle Routing Problem
(CVRP)**, the performance plots compare three different models across a
benchmark set. These plots offer a clear visual summary of how closely each
model approaches the best solution.

|                                                                                                                                 ![Performance Plot Objective](https://github.com/d-krupke/cpsat-primer/blob/main/images/performance_plot_objective.png?raw=true)                                                                                                                                  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Performance plot comparing the objective values of different CVRP models on a benchmark set. The Miller–Tucker–Zemlin model performs best on most instances and remains close to the best on the rest. The other two models find the best solution in only about 10% of instances but solve roughly 70% within 2% of the best known solution, with `multiple_circuit` showing a slight advantage. |

This can of course also be done for the lower bounds produced by each model.

|                                                                               ![Performance Plot Lower Bound](https://github.com/d-krupke/cpsat-primer/blob/main/images/performance_plot_bound.png?raw=true)                                                                               |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Performance plot comparing the lower bounds produced by each CVRP model. The `add_circuit` model consistently achieves the best bounds, while the other two models yield bounds that are up to 20% worse in the best case and up to 100% worse (i.e., half the quality) on some instances. |

<details>
<summary>Here is the code I used to generate the plots. You can freely copy and use it.</summary>

```python
# MIT License
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
```

</details>

> :reference:
>
> Tangi Migot has written an excellent article on
> [Performnace Plots](https://tmigot.github.io/posts/2024/06/teaching/)

### Analyzing the Scalability of a Single Model

When working with a single model and aiming to analyze its **scalability**, a
**split plot** can serve as an effective visualization. This type of plot shows
the model’s runtime across instances of varying size, under a fixed time limit.
If an instance is solved within the time limit, its actual runtime is shown. If
not, the point is plotted above the time limit on a **transformed y-axis** that
now displays the **optimality gap** instead of runtime.

An example of such a plot is shown below. Since aggregating this data into a
line plot can be challenging, the visualization may become cluttered if too many
instances are included or if multiple models are compared simultaneously.

|                                                                                                  ![Split Plot](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/split_plot.png)                                                                                                   |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Split plot illustrating runtime (for solved instances) and optimality gap (for unsolved instances). The y-axis is divided into two regions: one showing actual runtimes for instances solved within the time limit, and one showing optimality gaps for instances where the time limit was exceeded. |

> [!WARNING]
>
> For many problems, there is no single instance size metric to compare over.
> Usually, you can still classify the instances into size categories. However,
> for especially complex problems, it may be best to just provide a table with
> the results for the largest instances to give an idea of the model's
> scalability.

## A Complete Benchmarking Example

Designing an effective benchmark is a nuanced task that demands expertise. This
section aims to guide you in creating a reliable benchmark suitable for
publication purposes.

Given the breadth and complexity of benchmarking, our focus will be on the
basics, particularly through the lens of the Traveling Salesman Problem (TSP),
as previously discussed in the `add_circuit` section. We refer to the different
model implementations as 'solvers', and we'll explore four specific types:

- A solver employing the `add_circuit` approach.
- A solver based on the Miller-Tucker-Zemlin formulation.
- A solver utilizing the Dantzig-Fulkerson-Johnson formulation with iterative
  addition of subtour constraints until a connected tour is achieved.
- A Gurobi-based solver applying the Dantzig-Fulkerson-Johnson formulation via
  Lazy Constraints, which are not supported by CP-SAT.

This example highlights common challenges in benchmarking and strategies to
address them. A key obstacle in solving NP-hard problems is the variability in
solver performance across different instances. For instance, a solver might
easily handle a large instance but struggle with a smaller one, and vice versa.
Consequently, it is crucial to ensure that your benchmark encompasses a
representative variety of instances. This diversity is vital for drawing
meaningful conclusions, such as the maximum size of a TSP instance that can be
solved or the most effective solver to use.

For a comprehensive exploration of benchmarking, I highly recommend Catherine C.
McGeoch's book,
["A Guide to Experimental Algorithmics"](https://www.cambridge.org/core/books/guide-to-experimental-algorithmics/CDB0CB718F6250E0806C909E1D3D1082),
which offers an in-depth discussion on this topic.

### Distinguishing Exploratory and Workhorse Studies in Benchmarking

Before diving into comprehensive benchmarking, it is essential to conduct
preliminary investigations to assess your model’s capabilities and identify any
foundational issues. This phase, known as _exploratory studies_, is crucial for
establishing the basis for more detailed benchmarking, subsequently termed as
_workhorse studies_. These latter studies aim to provide reliable answers to
specific research questions and are often the core of academic publications. It
is important to explicitly differentiate between these two study types and
maintain their distinct purposes: exploratory studies for initial understanding
and flexibility, and workhorse studies for rigorous, reproducible research.

#### Exploratory Studies: Foundation Building

Exploratory studies serve as an introduction to both your model and the problem
it addresses. This phase is about gaining preliminary understanding and
insights.

- **Objective**: The goal here is to gather early insights rather than
  definitive conclusions. This phase is instrumental in identifying realistic
  problem sizes, potential challenges, and narrowing down hyperparameter search
  spaces.

For instance, in the `add_circuit`-section, an exploratory study helped us
determine that our focus should be on instances with 100 to 200 nodes. If you
encounter fundamental issues with your model at this stage, it’s advisable to
address these before proceeding to workhorse studies.

> [!WARNING]
>
> Occasionally, the primary performance bottleneck in your model may not be
> CP-SAT but rather the Python segment where the model is being generated. In
> these instances, identifying the most resource-intensive parts of your Python
> code is crucial. I have found the profiler
> [Scalene](https://github.com/plasma-umass/scalene) to be well-suited to
> investigate and pinpoint these bottlenecks.

#### Workhorse Studies: Conducting In-depth Evaluations

Workhorse studies follow the exploratory phase, characterized by more structured
and meticulous approaches. This stage is vital for a comprehensive evaluation of
your model and collecting substantive data for analysis.

- **Objective**: These studies are designed to answer specific research
  questions and provide meaningful insights. The approach here is more
  methodical, focusing on clearly defined research questions. The benchmarks
  designed should be well-structured and large enough to yield statistically
  significant results.

Remember, the aim is not to create a flawless benchmark right away but to evolve
it as concrete questions emerge and as your understanding of the model and
problem deepens. These studies, unlike exploratory ones, will be the focus of
your scientific publications, with exploratory studies only referenced for
justifying certain design decisions.

> [!TIP]
>
> Use the
> [SIGPLAN Empirical Evaluation Checklist](https://raw.githubusercontent.com/SIGPLAN/empirical-evaluation/master/checklist/checklist.pdf)
> if your evaluation has to satisfy academic standards.

### Designing a Robust Benchmark for Effective Studies

When undertaking both exploratory and workhorse studies, the creation of a
well-designed benchmark is a critical step. This benchmark is the basis upon
which you'll test and evaluate your solvers. For exploratory studies, your
benchmark can start simple and progressively evolve. However, when it comes to
workhorse studies, the design of your benchmark demands meticulous attention to
ensure comprehensiveness and reliability.

While exploratory studies also benefit from a thoughtfully designed benchmark—as
it accelerates insight acquisition—the primary emphasis at this stage is to have
a functioning benchmark in place. This initial benchmark acts as a springboard,
providing a foundation for deeper, more detailed analysis in the subsequent
workhorse studies. The key is to balance the immediacy of starting with a
benchmark against the long-term goal of refining it for more rigorous
evaluations.

Ideally, a robust benchmark would consist of a large set of real-world
instances, closely reflecting the actual performance of your solver. Real-world
instances, however, are often limited in quantity and may not provide enough
data for a statistically significant benchmark. In such cases, it is advisable
to explore existing benchmarks from literature, like the
[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) for TSP.
Leveraging established benchmarks allows for comparison with prior studies, but
be cautious about their quality, as not all are equally well-constructed. For
example, TSPLIB's limitations in terms of instance size variation and
heterogeneity can hinder result aggregation.

Therefore, creating custom instances might be necessary. When doing so, aim for
enough instances per size category to establish reliable and statistically
significant data points. For instance, generating 10 instances for each size
category (e.g., 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500) can
provide a solid basis for analysis. This approach, though modest in scale,
suffices to illustrate the benchmarking process.

Exercise caution with random instance generators, as they may not accurately
represent real-world scenarios. For example, randomly generated TSP instances
might lack collinear points common in real-world situations, like houses aligned
on straight roads, or they might not replicate real-world clustering patterns.
To better mimic reality, incorporate real-world data or use diverse generation
methods to ensure a broader variety of instances. For the TSP, we could for
example also have sampled from the larger TSPLIB instances.

Consider conducting your evaluation using two distinct benchmarks, especially
when dealing with different data types. For instance, you might have one
benchmark derived from real-world data which, although highly relevant, is too
limited in size to provide robust statistical insights. Simultaneously, you
could use a second benchmark based on a larger set of random instances, better
suited for detailed statistical analysis. This dual-benchmark approach allows
you to demonstrate the consistency and reliability of your results, ensuring
they are not merely artifacts of a particular dataset's characteristics. It's a
strategy that adds depth to your evaluation, showcasing the robustness of your
findings across varied data sources. We will use this approach below, generating
robust plots from random instances, but also comparing them to real-world
instances. Mixing the two benchmarks would not be advisable, as the random
instances would dominate the results.

Lastly, always separate the creation of your benchmark from the execution of
experiments. Create and save instances in a separate process to minimize errors.
The goal is to make your evaluation as error-proof as possible, avoiding the
frustration and wasted effort of basing decisions on flawed data. Be
particularly cautious with pseudo-random number generators; while theoretically
deterministic, their use can inadvertently lead to irreproducible results.
Sharing benchmarks is also more straightforward when you can distribute the
instances themselves, rather than the code used to generate them.

### Efficiently Managing Your Benchmarks

Managing benchmark data can become complex, especially with multiple experiments
and research questions. Here are some strategies to keep things organized:

- **Folder Structure**: Maintain a clear folder structure for your experiments,
  with a top-level `evaluations` folder and descriptive subfolders for each
  experiment. For our experiment we have the following structure:
  ```
  evaluations
  ├── tsp
  │   ├── 2023-11-18_random_euclidean
  │   │   ├── PRIVATE_DATA
  │   │   │   ├── ... all data for debugging
  │   │   ├── PUBLIC_DATA
  │   │   │   ├── ... selected data to share
  │   │   ├── README.md: Provide a short description of the experiment
  │   │   ├── 00_generate_instances.py
  │   │   ├── 01_run_experiments.py
  │   │   ├── ....
  │   ├── 2023-11-18_tsplib
  │   │   ├── PRIVATE_DATA
  │   │   │   ├── ... all data for debugging
  │   │   ├── PUBLIC_DATA
  │   │   │   ├── ... selected data to share
  │   │   ├── README.md: Provide a short description of the experiment
  │   │   ├── 01_run_experiments.py
  │   │   ├── ....
  ```
- **Redundancy and Documentation**: While some redundancy is acceptable,
  comprehensive documentation of each experiment is crucial for future
  reference.
- **Simplified Results**: Keep a streamlined version of your results for easy
  access, especially for plotting and sharing.
- **Data Storage**: Save all your data, even if it seems insignificant at the
  time. This ensures you have a comprehensive dataset for later analysis or
  unexpected inquiries. Because this can become a lot of data, it is advisable
  to have two folders: One with all data and one with a selection of data that
  you want to share.
- **Experiment Flexibility**: Design experiments to be interruptible and
  extendable, allowing for easy resumption or modification. This is especially
  important for exploratory studies, where you may need to make frequent
  adjustments. However, if your workhorse study takes a long time to run, you do
  not want to repeat it from scratch if you want to add a further solver.
- **Utilizing Technology**: Employ tools like slurm for efficient distribution
  of experiments across computing clusters, saving time and resources. The
  faster you have your results, the faster you can act on them.

Due to a lack of tools that exactly fitted my needs I developed
[AlgBench](https://github.com/d-krupke/AlgBench) to manage the results, and
[Slurminade](https://github.com/d-krupke/slurminade) to easily distribute the
experiments on a cluster via a simple decorator. However, there may be better
tools out there, now, especially from the Machine Learning community. Drop me a
quick mail if you have found some tools you are happy with, and I will take a
look myself.

### Analyzing the results

Let us now come to the actual analysis of the results. We will focus on the
following questions:

- Up to which size can we solve TSP instances with the different solvers?
- Which solver is the fastest?
- How does the performance change if we increase the optimality tolerance?

**Our Benchmarks:** We executed the four solvers with a time limit of 90s and
the optimality tolerances [0.1%, 1%, 5%, 10%, 25%] on a random benchmark set and
a TSPLIB benchmark set. The random benchmark set consists of 10 instances for
each number of nodes
`[25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]`. The weights were
chosen based on randomly embedding the nodes into a 2D plane and using the
Euclidean distances. The TSPLIB benchmark consists of all Euclidean instances
with less than 500 nodes. It is critical to have a time limit, as otherwise, the
benchmarks would take forever. You can find all find the whole experiment
[here](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/).

Let us first look at the results of the random benchmark, as they are easier to
interpret. We will then compare them to the TSPLIB benchmark.

#### Random Instances

A common, yet simplistic method to assess a model's performance involves
plotting its runtime against the size of the instances it processes. However,
this approach can often lead to inaccurate interpretations, particularly because
time-limited cutoffs can disproportionately affect the results. Instead of the
expected exponential curves, you will get skewed sigmoidal curves. Consequently,
such plots might not provide a clear understanding of the instance sizes your
model is capable of handling efficiently.

|                                                                             ![Runtime](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/runtime.png)                                                                              |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| The runtimes are sigmoidal instead of exponential because the time limit skews the results. The runtime can frequently exceed the time limit, because of expensive model building, etc. Thus, a pure runtime plot says surprisingly little (or is misleading) and can usually be discarded. |

Instead of just cutting off the runtime, a common metric is PAR10, which sets
the runtime to 10 times the time limit if the solver does not finish within the
time limit, and will actually penalize timeouts. However, it still does not
solve the problem that we actually do not know the true runtime such that these
plots will always lie.

To gain a more accurate insight into the capacities of your model, consider
plotting the proportion of instances of a certain size that your model
successfully solves. This method requires a well-structured benchmark to yield
meaningful statistics for each data point. Without this structure, the resulting
curve may appear erratic, making it challenging to draw dependable conclusions.

| ![Solved over size](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/solved_over_size.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: |
|                            For each x-value: What are the chances (y-values) that a model of this size (x) can be solved?                            |

Furthermore, if the pursuit is not limited to optimal solutions but extends to
encompass solutions of acceptable quality, the analysis can be expanded. One can
plot the number of instances that the model solves within a defined optimality
tolerance, as demonstrated in the subsequent figure:

| ![Solved over size with optimality tolerance](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/solved_over_size_opt_tol.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                              For each x-value: What are the chances (y-values) that a model of this size (x) can be solved to what quality (line style)?                               |

For a comparative analysis across various models against an arbitrary benchmark,
cactus plots emerge as a potent tool. These plots illustrate the number of
instances solved over time, providing a clear depiction of a model's efficiency.
For example, a coordinate of $x=10, y=20$ on such a plot signifies that 20
instances were solved within a span of 10 seconds each. It is important to note,
however, that these plots do not facilitate predictions for any specific
instance unless the benchmark set is thoroughly familiar. They do allow for an
estimation of which model is quicker for simpler instances and which can handle
more challenging instances within a reasonable timeframe. The question of what
exactly is a simple or challenging instance, however, is better answered by the
previous plots.

Cactus plots are notably prevalent in the evaluation of SAT-solvers, where
instance size is a poor indicator of difficulty. A more detailed discussion on
this subject can be found in the referenced academic paper:
[Benchmarking Solvers, SAT-style by Brain, Davenport, and Griggio](http://www.sc-square.org/CSA/workshop2-papers/RP3-FinalVersion.pdf)

| ![Cactus Plot 1](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/cactus_plot.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------: |
|                  For each x-value: How many (y) of the benchmark instances could have been solved with this time limit (x)?                  |

Additionally, the analysis can be refined to account for different quality
tolerances. This requires either multiple experimental runs or tracking the
progression of the lower and upper bounds within the solver. In the context of
CP-SAT, for instance, this tracking can be implemented via the Solution
Callback, although its activation is may depend on updates to the objective
rather than the bounds.

| ![Cactus Plot 1](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_random_euclidean/PUBLIC_DATA/cactus_plot_opt_tol.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: |
|    For each x-value: How many (y) of the benchmark instances could have been solved to a specific quality (line style) with this time limit (x)?     |

Instead of plotting the number of solved instances, one can also plot the number
of unsolved instances over time. This can be easier to read and additionally
indicates the number of instances in the benchmark. However, I personally do not
have a preference for one or the other, and would recommend using the one that
is more intuitive to read for you.

#### TSPLIB

Our second benchmark for the Traveling Salesman Problem leverages the TSPLIB, a
set of instances based on real-world data. This will introduce two challenges:

1. The difficulty in aggregating benchmark data due to its limited size and
   heterogeneous nature.
2. Notable disparities in results, arising from the differing characteristics of
   random and real-world instances.

The irregularity in instance sizes makes traditional plotting methods, like
plotting the number of solved instances over time, less effective. While data
smoothing methods, such as rolling averages, are available, they too have their
limitations.

| ![Variation in Data](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_tsplib/PUBLIC_DATA/solved_over_size.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------------: |
|       Such a plot may prove inefficient when dealing with high variability, particularly when some data points are underrepresented.        |

In contrast, the cactus plot still provides a clear and comprehensive
perspective of various model performances. An interesting observation we can
clearly see in it, is the diminished capability of the "Iterative Dantzig" model
in solving instances, and a closer performance alignment between the
`add_circuit` and Gurobi models.

| ![Effective Cactus Plot](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/tsp/2023-11-18_tsplib/PUBLIC_DATA/cactus_plot_opt_tol.png) |
| :------------------------------------------------------------------------------------------------------------------------------------------------: |
|                Cactus plots maintain clarity and relevance, and show a performance differences between TSPLib and random instances.                |

However, since cactus plots do not offer insights into individual instances, it
is beneficial to complement them with a detailed table of results for the
specific model you are focusing on. This approach ensures a more nuanced
understanding of model performance across varied instances. The following table
provides the results for the `add_circuit`-model.

| Instance | # nodes | runtime | lower bound | objective | opt. gap |
| :------- | ------: | ------: | ----------: | --------: | -------: |
| att48    |      48 |    0.47 |       33522 |     33522 |        0 |
| eil51    |      51 |    0.69 |         426 |       426 |        0 |
| st70     |      70 |     0.8 |         675 |       675 |        0 |
| eil76    |      76 |    2.49 |         538 |       538 |        0 |
| pr76     |      76 |   54.36 |      108159 |    108159 |        0 |
| kroD100  |     100 |    9.72 |       21294 |     21294 |        0 |
| kroC100  |     100 |    5.57 |       20749 |     20749 |        0 |
| kroB100  |     100 |     6.2 |       22141 |     22141 |        0 |
| kroE100  |     100 |    9.06 |       22049 |     22068 |        0 |
| kroA100  |     100 |    8.41 |       21282 |     21282 |        0 |
| eil101   |     101 |    2.24 |         629 |       629 |        0 |
| lin105   |     105 |    1.37 |       14379 |     14379 |        0 |
| pr107    |     107 |     1.2 |       44303 |     44303 |        0 |
| pr124    |     124 |    33.8 |       59009 |     59030 |        0 |
| pr136    |     136 |   35.98 |       96767 |     96861 |        0 |
| pr144    |     144 |   21.27 |       58534 |     58571 |        0 |
| kroB150  |     150 |   58.44 |       26130 |     26130 |        0 |
| kroA150  |     150 |   90.94 |       26498 |     26977 |       2% |
| pr152    |     152 |   15.28 |       73682 |     73682 |        0 |
| kroA200  |     200 |   90.99 |       29209 |     29459 |       1% |
| kroB200  |     200 |   31.69 |       29437 |     29437 |        0 |
| pr226    |     226 |   74.61 |       80369 |     80369 |        0 |
| gil262   |     262 |   91.58 |        2365 |      2416 |       2% |
| pr264    |     264 |   92.03 |       49121 |     49512 |       1% |
| pr299    |     299 |   92.18 |       47709 |     49217 |       3% |
| linhp318 |     318 |   92.45 |       41915 |     52032 |      19% |
| lin318   |     318 |   92.43 |       41915 |     52025 |      19% |
| pr439    |     439 |   94.22 |      105610 |    163452 |      35% |

This should highlight that often you need a combination of different benchmarks
and plots to get a good understanding of the performance of your model.
