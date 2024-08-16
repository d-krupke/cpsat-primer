<!-- EDIT THIS PART VIA 05_parameters.md -->

<a name="05-parameters"></a>

## Parameters

<!-- START_SKIP_FOR_README -->

![Cover Image Parameters](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_3.webp)

<!-- STOP_SKIP_FOR_README -->

The CP-SAT solver offers numerous parameters to control its behavior. These
parameters are implemented via
[Protocol Buffers](https://developers.google.com/protocol-buffers) and can be
manipulated using the `parameters` member. To explore all available options,
refer to the well-documented `proto` file in the
[official repository](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).
Below, I will highlight the most important parameters so you can get the most
out of CP-SAT.

> :warning: Only a few parameters, such as `max_time_in_seconds`, are suitable
> for beginners. Most other parameters, like decision strategies, are best left
> at their default settings, as they are well-chosen and tampering with them
> could disrupt optimizations. For better performance, focus on improving your
> model.

### Logging

The `log_search_progress` parameter is crucial at the beginning. It enables
logging of the search progress, providing insights into how CP-SAT solves your
problem. While you may deactivate it later for production, it is beneficial
during development to understand the process and respond to any issues.

```python
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True

# Custom log function, for example, using the Python logging module instead of stdout
# Useful in a Jupyter notebook, where logging to stdout might not be visible
solver.log_callback = print  # (str)->None
# If using a custom log function, you can disable logging to stdout
solver.parameters.log_to_stdout = False
```

The log offers valuable information for understanding CP-SAT and your
optimization problem. It details aspects such as how many variables were
directly removed and which techniques most effectively contributed to improving
lower and upper bounds.

An example log might look like this:

```
Starting CP-SAT solver v9.10.4067
Parameters: max_time_in_seconds: 30 log_search_progress: true relative_gap_limit: 0.01
Setting number of workers to 16

Initial optimization model '': (model_fingerprint: 0x1d316fc2ae4c02b1)
#Variables: 450 (#bools: 276 #ints: 6 in objective)
  - 342 Booleans in [0,1]
  - 12 in [0][10][20][30][40][50][60][70][80][90][100]
  - 6 in [0][10][20][30][40][100]
  - 6 in [0][80][100]
  - 6 in [0][100]
  - 6 in [0,1][34][67][100]
  - 12 in [0,6]
  - 18 in [0,7]
  - 6 in [0,35]
  - 6 in [0,36]
  - 6 in [0,100]
  - 12 in [21,57]
  - 12 in [22,57]
#kBoolOr: 30 (#literals: 72)
#kLinear1: 33 (#enforced: 12)
#kLinear2: 1'811
#kLinear3: 36
#kLinearN: 94 (#terms: 1'392)

Starting presolve at 0.00s
  3.26e-04s  0.00e+00d  [DetectDominanceRelations]
  6.60e-03s  0.00e+00d  [PresolveToFixPoint] #num_loops=4 #num_dual_strengthening=3
  2.69e-05s  0.00e+00d  [ExtractEncodingFromLinear] #potential_supersets=44 #potential_subsets=12
[Symmetry] Graph for symmetry has 2'224 nodes and 5'046 arcs.
[Symmetry] Symmetry computation done. time: 0.000374304 dtime: 0.00068988
[Symmetry] #generators: 2, average support size: 12
[Symmetry] 12 orbits with sizes: 2,2,2,2,2,2,2,2,2,2,...
[Symmetry] Found orbitope of size 6 x 2
[SAT presolve] num removable Booleans: 0 / 309
[SAT presolve] num trivial clauses: 0
[SAT presolve] [0s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
[SAT presolve] [3.0778e-05s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
[SAT presolve] [4.6758e-05s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
  1.10e-02s  9.68e-03d  [Probe] #probed=1'738 #new_bounds=12 #new_binary_clauses=1'111
  2.34e-03s  0.00e+00d  [MaxClique] Merged 602(1374 literals) into 506(1960 literals) at_most_ones.
  3.31e-04s  0.00e+00d  [DetectDominanceRelations]
  1.89e-03s  0.00e+00d  [PresolveToFixPoint] #num_loops=2 #num_dual_strengthening=1
  5.45e-04s  0.00e+00d  [ProcessAtMostOneAndLinear]
  8.19e-04s  0.00e+00d  [DetectDuplicateConstraints] #without_enforcements=306
  8.62e-05s  7.21e-06d  [DetectDominatedLinearConstraints] #relevant_constraints=114 #num_inclusions=42
  1.94e-05s  0.00e+00d  [DetectDifferentVariables]
  1.90e-04s  8.39e-06d  [ProcessSetPPC] #relevant_constraints=560 #num_inclusions=24
  2.01e-05s  0.00e+00d  [FindAlmostIdenticalLinearConstraints]
...
```

Given the complexity of the log, I developed a tool to visualize and comment on
it. You can copy and paste your log into the tool, which will automatically
highlight the most important details. Be sure to check out the examples.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cpsat-log-analyzer.streamlit.app/)
[![d-krupke - CP-SAT Log Analyzer](https://img.shields.io/badge/d--krupke-CP--SAT%20Log%20Analyzer-blue?style=for-the-badge&logo=github)](https://github.com/d-krupke/CP-SAT-Log-Analyzer)

|                                                                                                                       ![Search Progress](https://github.com/d-krupke/cpsat-primer/blob/main/images/search_progress.png)                                                                                                                       |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| A plot of the search progress over time as visualized by the log analyzer using information from the log (a different log than displayed above). This plot helps you understand which part of your problem is more challenging: finding a good solution or proving its quality. Based on this, you can implement appropriate countermeasures. |

We will revisit the logs in the next chapter.

> [!TIP]
>
> From my experience as a lecturer, I often encounter students who believe
> CP-SAT is stuck, only to discover that their model building includes an
> unnecessarily complex $O(n^5)$ nested loop, which would take days to run. It
> is natural to assume that the issue lies with CP-SAT because it handles the
> hard part of solving the problem. However, even the seemingly simple part of
> model building can consume a lot of time if implemented incorrectly. By
> enabling logging, students could immediately see that the issue lies in their
> own code rather than with CP-SAT. This simple step can save a lot of time and
> frustration.

### Time Limit and Status

When working with large or complex models, the CP-SAT solver may not always
reach an optimal solution within a reasonable time frame and could potentially
run indefinitely. Therefore, setting a time limit is advisable, particularly in
a production environment, to prevent the solver from running endlessly. Even
within a time limit, CP-SAT often finds a reasonably good solution, although it
may not be proven optimal.

Determining an appropriate time limit depends on various factors and usually
requires some experimentation. I typically start with a time limit between 60
and 300 seconds, as this provides a balance between not having to wait too long
during model testing and giving the solver enough time to find a good solution.

To set a time limit (in seconds) before running the solver, use the following
command:

```python
solver.parameters.max_time_in_seconds = 60  # 60s time limit
```

After running the solver, it is important to check the status to determine
whether an optimal solution, a feasible solution, or no solution at all has been
found:

```python
status = solver.solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("We have a solution.")
else:
    print("Help?! No solution available! :( ")
```

The possible status codes are:

- `OPTIMAL` (4): An optimal solution has been found.
- `FEASIBLE` (2): A feasible solution has been found, and a bound may be
  available to assess its quality via `solver.best_objective_bound`.
- `INFEASIBLE` (3): No solution can satisfy all constraints.
- `MODEL_INVALID` (1): The CP-SAT model is incorrectly specified.
- `UNKNOWN` (0): No solution was found, and no infeasibility proof is available.
  A bound may still be available.

To get the name from the status code, use `solver.status_name(status)`.

In addition to limiting runtime, you can specify acceptable solution quality
using `absolute_gap_limit` and `relative_gap_limit`. The absolute limit stops
the solver when the solution is within a specified value of the bound. The
relative limit stops the solver when the objective value (O) is within a
specified ratio of the bound (B). To stop when the solution is (provably) within
5% of the optimum, use:

```python
solver.parameters.relative_gap_limit = 0.05
```

For cases where progress stalls or for other reasons, solution callbacks can be
used to halt the solver. With these, you can decide on every new solution if the
solution is good enough or if the solver should continue searching for a better
one. Unlike Gurobi, CP-SAT does not support adding lazy constraints from these
callbacks (or at all), which is a significant limitation for problems requiring
dynamic model adjustments.

To add a solution callback, inherit from the base class
`CpSolverSolutionCallback`. Documentation for this base class and its operations
is available
[here](https://developers.google.com/optimization/reference/python/sat/python/cp_model#cp_model.CpSolverSolutionCallback).

```python
class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, data):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.data = data  # Store data in the callback.

    def on_solution_callback(self):
        obj = self.objective_value  # Best solution value
        bound = self.best_objective_bound  # Best bound
        print(f"The current value of x is {self.Value(x)}")
        if abs(obj - bound) < 10:
            self.StopSearch()  # Stop search for a better solution
        # ...


solver.solve(model, MySolutionCallback(None))
```

An
[official example using callbacks](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py)
is available.

Besides querying the objective value of the best solution and the best known
bound, you can also access internal metrics such as `self.num_booleans`,
`self.num_branches`, and `self.num_conflicts`. These metrics will be discussed
later.

As of version 9.10, CP-SAT also supports bound callbacks, which are triggered
when the proven bound improves. Unlike solution callbacks, which activate upon
finding new solutions, bound callbacks are useful for stopping the search when
the bound is sufficiently good. The syntax for bound callbacks differs from that
of solution callbacks, as they are implemented as free functions that directly
access the solver object.

```python
solver = cp_model.CpSolver()


def bound_callback(bound):
    print(f"New bound: {bound}")
    if bound > 100:
        solver.stop_search()


solver.best_bound_callback = bound_callback
```

Instead of using a simple function, you can also use a callable object to store
a reference to the solver object. This approach allows you to define the
callback outside the local scope, providing greater flexibility.

```python
class BoundCallback:
    def __init__(self, solver) -> None:
        self.solver = solver

    def __call__(self, bound):
        print(f"New bound: {bound}")
        if bound > 200:
            print("Abort search due to bound")
            self.solver.stop_search()
```

This method is more flexible than the solution callback and can be considered
more Pythonic.

Additionally, whenever there is a new solution or bound, a log message is
generated. You can hook into the log output to decide when to stop the search
using CP-SAT's log callback.

```python
solver.parameters.log_search_progress = True  # Enable logging
solver.log_callback = lambda msg: print("LOG:", msg)  # (str) -> None
```

### Parallelization

CP-SAT is a portfolio-solver that uses different techniques to solve the
problem. There is some information exchange between the different workers, but
it does not split the solution space into different parts, thus, it does not
parallelize the branch-and-bound algorithm as MIP-solvers do. This can lead to
some redundancy in the search, but running different techniques in parallel will
also increase the chance of running the right technique. Predicting which
technique will be the best for a specific problem is often hard, thus, this
parallelization can be quite useful.

By default, CP-SAT leverages all available cores. You can control the
parallelization of CP-SAT by setting the number of search workers.

```python
solver.parameters.num_workers = 8  # use 8 cores
```

Here are the solvers used by CP-SAT 9.9 on different parallelization levels for
an optimization problem and no additional specifications (e.g., decision
strategies). Note that some parameters/constraints/objectives can change the
parallelization strategy. Also check
[the official documentation](https://github.com/google/or-tools/blob/main/ortools/sat/docs/troubleshooting.md#improving-performance-with-multiple-workers).

| # Workers | Full Problem Subsolvers                                                             | First Solution Subsolvers                                                                                                                                                                                                                                                                      | Incomplete Subsolvers                                                                                                                                                                                                                                                       | Helper Subsolvers                                                                      |
| --------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **1**     | `default_lp`                                                                        | -                                                                                                                                                                                                                                                                                              | -                                                                                                                                                                                                                                                                           | -                                                                                      |
| **2**     | No change                                                                           | No change                                                                                                                                                                                                                                                                                      | Added 13 solvers: `feasibility_pump`, `graph_arc_lns`, `graph_cst_lns`, `graph_dec_lns`, `graph_var_lns`, `packing_precedences_lns`, `packing_rectangles_lns`, `packing_slice_lns`, `rins/rens`, `rnd_cst_lns`, `rnd_var_lns`, `scheduling_precedences_lns`, `violation_ls` | Added 3 solvers: `neighborhood_helper`, `synchronization_agent`, `update_gap_integral` |
| **3**     | Added 1 solver: `no_lp`                                                             | No change                                                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **4**     | Added 1 solver: `max_lp`                                                            | No change                                                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **5**     | No change                                                                           | Added 1 solver: `fj_short_default`                                                                                                                                                                                                                                                             | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **6**     | Added 1 solver: `quick_restart`                                                     | No change                                                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **7**     | Added 1 solver: `reduced_costs`                                                     | No change                                                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **8**     | Added 1 solver: `quick_restart_no_lp`                                               | No change                                                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **12**    | Added 2 solvers: `lb_tree_search`, `pseudo_costs`                                   | Added 2 solvers: `fj_long_default`, `fs_random`                                                                                                                                                                                                                                                | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **16**    | Added 3 solvers: `objective_lb_search`, `objective_shaving_search_no_lp`, `probing` | Added 1 solver: `fs_random_quick_restart`                                                                                                                                                                                                                                                      | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **20**    | Added 2 solvers: `objective_shaving_search_max_lp`, `probing_max_lp`                | Added 1 solver: `fj_short_lin_default`                                                                                                                                                                                                                                                         | No change                                                                                                                                                                                                                                                                   | No change                                                                              |
| **32**    | Added 3 solvers: `objective_lb_search_max_lp`, `objective_lb_search_no_lp`          | Added 8 solvers: `fj_long_lin_default`, `fj_long_lin_random`, `fj_long_random`, `fj_short_lin_random`, `fj_short_random`, `fs_random_no_lp`, `fs_random_quick_restart_no_lp`                                                                                                                   | No change                                                                                                                                                                                                                                                                   |
| **64**    | No change                                                                           | Added 11 solvers: `fj_long_default(2)`, `fj_long_lin_default(2)`, `fj_long_lin_random(2)`, `fj_long_random(2)`, `fj_short_default(2)`, `fj_short_lin_default(2)`, `fj_short_random(2)`, `fs_random(6)`, `fs_random_no_lp(6)`, `fs_random_quick_restart(6)`, `fs_random_quick_restart_no_lp(5)` | Added 1 solver: `violation_ls(7)`                                                                                                                                                                                                                                           |

Important steps:

- With a single worker, only the default subsolver is used.
- With two workers or more, CP-SAT starts using incomplete subsolvers, i.e.,
  heuristics such as LNS.
- With five workers, CP-SAT will also have a first solution subsolver.
- With 23 workers, all 15 full problem subsolvers are used.
- For more than 32 workers, primarily the number of first solution subsolvers is
  increased.

**Full problem subsolvers** are solvers that search the full problem space,
e.g., by a branch-and-bound algorithm. Available full problem subsolvers are:

- `default_lp`: LCG-based search with default linearization of the model.
  - `max_lp`: Same as `default_lp` but with maximal linearization.
  - `no_lp`: Same as `default_lp` but without linearization.
- `lb_tree_search`: This solver is focussed on improving the proven bound, not
  on finding better solutions. By disproving the feasibility of the cheapest
  nodes in the search tree, it incrementally improves the bound, but has only
  little chances to find better solutions.
- `objective_lb_search`: Also focussed on improving the bound by disproving the
  feasibility of the current lower bound.
  - `objective_lb_search_max_lp`: With maximal linearization.
  - `objective_lb_search_no_lp`: Without linearization.
  - `objective_shaving_search_max_lp`: Should be quite similar to
    `objective_lb_search_max_lp`.
  - `objective_shaving_search_no_lp`: Should be quite similar to
    `objective_lb_search_no_lp`.
- `probing`: Fixing variables and seeing what happens.
  - `probing_max_lp`: Same as probing but with maximal linearization.
- `pseudo_costs`: Uses pseudo costs for branching, which are computed from
  historical changes in objective bounds following certain branching decisions.
- `quick_restart`: Restarts the search more eagerly. Restarts rebuild the search
  tree from scratch, but keep learned clauses. This allows to recover from bad
  decisions, and lead to smaller search trees by learning from the mistakes of
  the past.
  - `quick_restart_no_lp`: Same as `quick_restart` but without linearization.
- `reduced_costs`: Uses the reduced costs of the linear relaxation for
  branching.
- `core`: A strategy from the SAT-community that extracts unsatisfiable cores of
  the formula.
- `fixed`: User-specified search strategy.

You can modify the used subsolvers by `solver.parameters.subsolvers`,
`solver.parameters.extra_subsolvers`, and `solver.parameters.ignore_subsolvers`.
This can be interesting, e.g., if you are using CP-SAT especially because the
linear relaxation is not useful (and the BnB-algorithm performing badly). There
are even more options, but for these you can simply look into the
[documentation](https://github.com/google/or-tools/blob/49b6301e1e1e231d654d79b6032e79809868a70e/ortools/sat/sat_parameters.proto#L513).
Be aware that fine-tuning such a solver is not a simple task, and often you do
more harm than good by tinkering around. However, I noticed that decreasing the
number of search workers can actually improve the runtime for some problems.
This indicates that at least selecting the right subsolvers that are best fitted
for your problem can be worth a shot. For example `max_lp` is probably a waste
of resources if you know that your model has a terrible linear relaxation. In
this context I want to recommend having a look on some relaxed solutions when
dealing with difficult problems to get a better understanding of which parts a
solver may struggle with (use a linear programming solver, like Gurobi, for
this).

You can evaluate the performance of the different strategies by looking at the
`Solutions` and `Objective bounds` blocks in the log. Here an example:

```
Solutions (7)             Num   Rank
                'no_lp':    3  [1,7]
        'quick_restart':    1  [3,3]
  'quick_restart_no_lp':    3  [2,5]

Objective bounds                     Num
                  'initial_domain':    1
             'objective_lb_search':    2
       'objective_lb_search_no_lp':    4
  'objective_shaving_search_no_lp':    1
```

For solutions, the first number is the number of solutions found by the
strategy, the second number is the range of the ranks of the solutions. The
value `[1,7]` indicates that the solutions found by the strategy have ranks
between 1 and 7. In this case, it means that the strategy `no_lp` found the best
and the worst solution.

For objective bounds, the number indicates how often the strategy contributed to
the best bound. For this example, it seems that the `no_lp` strategies are the
most successful. Note that for both cases, it is more interesting, which
strategies do not appear in the list.

In the search log, you can also see at which time which subsolver contributed
something. This log also includes the incomplete and first solution subsolvers.

```
#1       0.01s best:43    next:[6,42]     no_lp (fixed_bools=0/155)
#Bound   0.01s best:43    next:[7,42]     objective_shaving_search_no_lp (vars=73 csts=120)
#2       0.01s best:33    next:[7,32]     quick_restart_no_lp (fixed_bools=0/143)
#3       0.01s best:31    next:[7,30]     quick_restart (fixed_bools=0/123)
#4       0.01s best:17    next:[7,16]     quick_restart_no_lp (fixed_bools=2/143)
#5       0.01s best:16    next:[7,15]     quick_restart_no_lp (fixed_bools=22/147)
#Bound   0.01s best:16    next:[8,15]     objective_lb_search_no_lp
#6       0.01s best:15    next:[8,14]     no_lp (fixed_bools=41/164)
#7       0.01s best:14    next:[8,13]     no_lp (fixed_bools=42/164)
#Bound   0.01s best:14    next:[9,13]     objective_lb_search
#Bound   0.02s best:14    next:[10,13]    objective_lb_search_no_lp
#Bound   0.04s best:14    next:[11,13]    objective_lb_search_no_lp
#Bound   0.06s best:14    next:[12,13]    objective_lb_search
#Bound   0.25s best:14    next:[13,13]    objective_lb_search_no_lp
#Model   0.26s var:125/126 constraints:162/162
#Model   2.24s var:124/126 constraints:160/162
#Model   2.58s var:123/126 constraints:158/162
#Model   2.91s var:121/126 constraints:157/162
#Model   2.95s var:120/126 constraints:155/162
#Model   2.97s var:109/126 constraints:140/162
#Model   2.98s var:103/126 constraints:135/162
#Done    2.98s objective_lb_search_no_lp
#Done    2.98s quick_restart_no_lp
#Model   2.98s var:66/126 constraints:91/162
```

**Incomplete subsolvers** are solvers that do not search the full problem space,
but work heuristically. Notable strategies are large neighborhood search (LNS)
and feasibility pumps. The first one tries to find a better solution by changing
only a few variables, the second one tries to make infeasible/incomplete
solutions feasible. You can also run only the incomplete subsolvers by setting
`solver.parameters.use_lns_only = True`, but this needs to be combined with a
time limit, as the incomplete subsolvers do not know when to stop.

**First solution subsolvers** are strategies that try to find a first solution
as fast as possible. They are often used to warm up the solver and to get a
first impression of the problem.

<!-- Source on Parallelization in Gurobi and general opportunities -->

If you are interested in how Gurobi parallelizes its search, you can find a
great video [here](https://www.youtube.com/watch?v=FJz1UxaMWRQ). Ed Rothberg
also explains the general opportunities and challenges of parallelizing a
solver, making it also interesting for understanding the parallelization of
CP-SAT.

<!-- Give a disclaimer -->

> :warning: This section could need some help as there is the possibility that I
> am mixing up some of the strategies, or am drawing wrong connections.

#### Importing/Exporting Models for Comparison on Different Hardware

If you want to compare the performance of different parallelization levels or
different hardware, you can use the following code snippets to export a model.
Instead of having to rebuild the model or share any code, you can then simply
load the model on a different machine and run the solver.

```python
from ortools.sat.python import cp_model
from google.protobuf import text_format


def export_model(model: cp_model.CpModel, filename: str):
    with open(filename, "w") as file:
        file.write(text_format.MessageToString(model.Proto()))


def import_model(filename: str) -> cp_model.CpModel:
    model = cp_model.CpModel()
    with open(filename, "r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model
```

### Assumptions

Often, you may need to explore the impact of forcing certain variables to
specific values. To avoid copying the entire model multiple times, CP-SAT offers
a convenient option: adding assumptions. Assumptions can be cleared afterward,
allowing you to test new assumptions without duplicating the model. This
feature, common in many SAT solvers, is restricted to boolean literals in
CP-SAT. While you cannot add complex expressions directly, using auxiliary
boolean variables enables you to experiment with more intricate constraints
without model duplication. For temporary complex constraints, model copying
using `model.CopyFrom` may still be necessary, along with variable copying.

```python
b1 = model.new_bool_var("b1")
b2 = model.new_bool_var("b2")
b3 = model.new_bool_var("b3")

model.add_assumptions([b1, ~b2])  # assume b1=True, b2=False
model.add_assumption(b3)  # assume b3=True (single literal)
# ... solve again and analyze ...
model.clear_assumptions()  # clear all assumptions
```

> [!NOTE]
>
> While incremental SAT solvers can reuse learned clauses from previous runs
> despite changing assumptions, CP-SAT does not support this feature.
> Assumptions in CP-SAT only help avoid model duplication.

### Hints

If you have a good intuition about how the solution might look—perhaps from
solving a similar model or using a good heuristic—you can inform CP-SAT to
incorporate this knowledge into its search. Some workers will try to follow
these hints, which can significantly improve the solver's performance if they
are good. If the hints actually represent a feasible solution, the solver can
use them to prune the search space of all branches that have worse bounds than
the hints.

```python
model.add_hint(x, 1)  # Suggest that x will probably be 1
model.add_hint(y, 2)  # Suggest that y will probably be 2
```

For more examples, refer to
[the official example](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/solution_hinting_sample_sat.py).
We will also see how to utilize hints for multi-objective optimization in the
next chapter.

> [!TIP]
>
> Hints can significantly improve solver performance, especially if it struggles
> to find a good initial solution (as indicated in the logs). This practice is
> often called **warm-starting** the solver. You do not need to provide values
> for all auxiliary variables, but if you use integer variables to approximate
> continuous variables, it is beneficial to provide hints for these. CP-SAT may
> struggle with quickly completing the solution, and only completed solutions
> can be used for pruning the search space. If CP-SAT needs a long time to
> complete the solution from the hint, it may have wasted a lot of time in
> branches it could otherwise have pruned.

To ensure your hints are correct, you can enable the following parameter, which
will make CP-SAT throw an error if the hints are incorrect:

```python
solver.parameters.debug_crash_on_bad_hint = True
```

If you suspect that your hints are not being utilized, it might indicate a
logical error in your model or a bug in your code. This parameter can help
diagnose such issues. However, this feature does not work reliably, so it should
not be solely relied upon.

> [!WARNING]
>
> In older versions of CP-SAT, hints could sometimes visibly slow down the
> solver, even if they were correct but not optimal. While this issue seems
> resolved in the latest versions, it is important to note that bad hints can
> still cause slowdowns by guiding the solver in the wrong direction.

### Decision Strategy

In the end of this section, a more advanced parameter that looks interesting for
advanced users as it gives some insights into the search algorithm. It can be
useful in combination with `solver.parameters.enumerate_all_solutions = True` to
specify the order in which all solutions are iterated. It can also have some
impact on the search performance for normal optimization, but this is often hard
to predict, thus, you should leave the following parameters unless you have a
good reason to change them.

We can tell CP-SAT, how to branch (or make a decision) whenever it can no longer
deduce anything via propagation. For this, we need to provide a list of the
variables (order may be important for some strategies), define which variable
should be selected next (fixed variables are automatically skipped), and define
which value should be probed.

We have the following options for variable selection:

- `CHOOSE_FIRST`: the first not-fixed variable in the list.
- `CHOOSE_LOWEST_MIN`: the variable that could (potentially) take the lowest
  value.
- `CHOOSE_HIGHEST_MAX`: the variable that could (potentially) take the highest
  value.
- `CHOOSE_MIN_DOMAIN_SIZE`: the variable that has the fewest feasible
  assignments.
- `CHOOSE_MAX_DOMAIN_SIZE`: the variable the has the most feasible assignments.

For the value/domain strategy, we have the options:

- `SELECT_MIN_VALUE`: try to assign the smallest value.
- `SELECT_MAX_VALUE`: try to assign the largest value.
- `SELECT_LOWER_HALF`: branch to the lower half.
- `SELECT_UPPER_HALF`: branch to the upper half.
- `SELECT_MEDIAN_VALUE`: try to assign the median value.

```python
model.add_decision_strategy([x], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

# your can force CP-SAT to follow this strategy exactly
solver.parameters.search_branching = cp_model.FIXED_SEARCH
```

For example for [coloring](https://en.wikipedia.org/wiki/Graph_coloring) (with
integer representation of the color), we could order the variables by decreasing
neighborhood size (`CHOOSE_FIRST`) and then always try to assign the lowest
color (`SELECT_MIN_VALUE`). This strategy should perform an implicit
kernelization, because if we need at least $k$ colors, the vertices with less
than $k$ neighbors are trivial (and they would not be relevant for any
conflict). Thus, by putting them at the end of the list, CP-SAT will only
consider them once the vertices with higher degree could be colored without any
conflict (and then the vertices with lower degree will, too). Another strategy
may be to use `CHOOSE_LOWEST_MIN` to always select the vertex that has the
lowest color available. Whether this will actually help, has to be evaluated:
CP-SAT will probably notice by itself which vertices are the critical ones after
some conflicts.

> :warning: I played around a little with selecting a manual search strategy.
> But even for the coloring, where this may even seem smart, it only gave an
> advantage for a bad model and after improving the model by symmetry breaking,
> it performed worse. Further, I assume that CP-SAT can learn the best strategy
> (Gurobi does such a thing, too) much better dynamically on its own.

---
