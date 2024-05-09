<!-- EDIT THIS PART VIA 05_parameters.md -->

<a name="05-parameters"></a>

## Parameters

![Cover Image Parameters](./images/logo_3.webp)

The CP-SAT solver has a lot of parameters to control its behavior. They are
implemented via
[Protocol Buffer](https://developers.google.com/protocol-buffers) and can be
manipulated via the `parameters`-member. If you want to find out all options,
you can check the reasonably well documented `proto`-file in the
[official repository](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).
I will give you the most important right below.

> :warning: Only a few of the parameters (e.g., timelimit) are
> beginner-friendly. Most other parameters (e.g., decision strategies) should
> not be touched as the defaults are well-chosen, and it is likely that you will
> interfere with some optimizations. If you need a better performance, try to
> improve your model of the optimization problem.

### Time limit and Status

If we have a huge model, CP-SAT may not be able to solve it to optimality (if
the constraints are not too difficult, there is a good chance we still get a
good solution). Of course, we do not want CP-SAT to run endlessly for hours
(years, decades,...) but simply abort after a fixed time and return us the best
solution so far. If you are now asking yourself why you should use a tool that
may run forever: There are simply no provably faster algorithms and considering
the combinatorial complexity, it is incredible that it works so well. Those not
familiar with the concepts of NP-hardness and combinatorial complexity, I
recommend reading the book 'In Pursuit of the Traveling Salesman' by William
Cook. Actually, I recommend this book to anyone into optimization: It is a
beautiful and light weekend-read.

To set a time limit (in seconds), we can simply set the following value before
we run the solver:

```python
solver.parameters.max_time_in_seconds = 60  # 60s timelimit
```

We now of course have the problem, that maybe we will not have an optimal
solution, or a solution at all, we can continue on. Thus, we need to check the
status of the solver.

```python
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("We have a solution.")
else:
    print("Help?! No solution available! :( ")
```

The following status codes are possible:

- `OPTIMAL`: Perfect, we have an optimal solution.
- `FEASIBLE`: Good, we have at least a feasible solution (we may also have a
  bound available to check the quality via `solver.BestObjectiveBound()`).
- `INFEASIBLE`: There is a proof that no solution can satisfy all our
  constraints.
- `MODEL_INVALID`: You used CP-SAT wrongly.
- `UNKNOWN`: No solution was found, but also no infeasibility proof. Thus, we
  actually know nothing. Maybe there is at least a bound available?

If you want to print the status, you can use `solver.StatusName(status)`.

We can not only limit the runtime but also tell CP-SAT, we are satisfied with a
solution within a specific, provable quality range. For this, we can set the
parameters `absolute_gap_limit` and `relative_gap_limit`. The absolute limit
tells CP-SAT to stop as soon as the solution is at most a specific value apart
to the bound, the relative limit is relative to the bound. More specifically,
CP-SAT will stop as soon as the objective value(O) is within relative ratio
$abs(O - B) / max(1, abs(O))$ of the bound (B). To stop as soon as we are within
5% of the optimum, we could state (TODO: Check)

```python
solver.parameters.relative_gap_limit = 0.05
```

Now we may want to stop after we did not make progress for some time or
whatever. In this case, we can make use of the solution callbacks.

> For those familiar with Gurobi: Unfortunately, we can only abort the solution
> progress and not add lazy constraints or similar. For those not familiar with
> Gurobi or MIPs: With Mixed Integer Programming we can adapt the model during
> the solution process via callbacks which allows us to solve problems with
> many(!) constraints by only adding them lazily. This is currently the biggest
> shortcoming of CP-SAT for me. Sometimes you can still do dynamic model
> building with only little overhead feeding information of previous iterations
> into the model

For adding a solution callback, we have to inherit from a base class. The
documentation of the base class and the available operations can be found in the
[documentation](https://developers.google.com/optimization/reference/python/sat/python/cp_model#cp_model.CpSolverSolutionCallback).

```python
class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, stuff):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.stuff = stuff  # just to show that we can save some data in the callback.

    def on_solution_callback(self):
        obj = self.ObjectiveValue()  # best solution value
        bound = self.BestObjectiveBound()  # best bound
        print(f"The current value of x is {self.Value(x)}")
        if abs(obj - bound) < 10:
            self.StopSearch()  # abort search for better solution
        # ...


solver.Solve(model, MySolutionCallback(None))
```

You can find an
[official example of using such callbacks](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py)
.

Beside querying the objective value of the currently best solution, the solution
itself, and the best known bound, you can also find out about internals such as
`NumBooleans(self)`, `NumConflicts(self)`, `NumBranches(self)`. What those
values mean will be discussed later.

With version 9.10, CP-SAT also supports bound callbacks. These are called when
CP-SAT is able to improve the proven bound. As the solution callback is only
called when a new solution is found, you may want to additionally use the bound
callback when you want to abort the search as soon as the bound is good enough.

```python
solver = cp_model.CpSolver()


def bound_callback(bound):
    print(f"New bound: {bound}")
    if bound > 100:
        solver.StopSearch()


solver.best_bound_callback = bound_callback
```

You can also try to directly hook into the log output of CP-SAT, which should be
called for new solutions, new bounds, and other important events. This can be
done by setting the `log_callback` parameter.

```python
solver.log_callback = lambda msg: print("LOG:", msg)  # (str)->None
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

You can control the parallelization of CP-SAT by setting the number of search
workers.

```python
solver.parameters.num_search_workers = 8  # use 8 cores
```

Here the solvers used by CP-SAT 9.9 on different parallelization levels for an
optimization problem and no additional specifications (e.g., decision
strategies). Note that some parameters/constraints/objectives can change the
parallelization strategy Also check
[the official documentation](https://github.com/google/or-tools/blob/main/ortools/sat/docs/troubleshooting.md#improving-performance-with-multiple-workers).

- `solver.parameters.num_search_workers = 1`: Single-threaded search with
  `[default_lp]`.
  - 1 full problem subsolver: [default_lp]
- `solver.parameters.num_search_workers = 2`: Additional use of heuristics to
  support the `default_lp` search.
  - 1 full problem subsolver: [default_lp]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 3`: Using a second full problem solver
  that does not try to linearize the model.
  - 2 full problem subsolvers: [default_lp, no_lp]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 4`: Additionally using a third full
  problem solver that tries to linearize the model as much as possible.
  - 3 full problem subsolvers: [default_lp, max_lp, no_lp]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 5`: Additionally using a first
  solution subsolver.
  - 3 full problem subsolvers: [default_lp, max_lp, no_lp]
  - 1 first solution subsolver: [fj_short_default]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 6`: Using a fourth full problem solver
  `quick_restart` that does more "probing".
  - 4 full problem subsolvers: [default_lp, max_lp, no_lp, quick_restart]
  - 1 first solution subsolver: [fj_short_default]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 7`:
  - 5 full problem subsolvers: [default_lp, max_lp, no_lp, quick_restart,
    reduced_costs]
  - 1 first solution subsolver: [fj_short_default]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 8`:
  - 6 full problem subsolvers: [default_lp, max_lp, no_lp, quick_restart,
    quick_restart_no_lp, reduced_costs]
  - 1 first solution subsolver: [fj_short_default]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 12`:
  - 8 full problem subsolvers: [default_lp, lb_tree_search, max_lp, no_lp,
    pseudo_costs, quick_restart, quick_restart_no_lp, reduced_costs]
  - 3 first solution subsolvers: [fj_long_default, fj_short_default, fs_random]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 16`:
  - 11 full problem subsolvers: [default_lp, lb_tree_search, max_lp, no_lp,
    objective_lb_search, objective_shaving_search_no_lp, probing, pseudo_costs,
    quick_restart, quick_restart_no_lp, reduced_costs]
  - 4 first solution subsolvers: [fj_long_default, fj_short_default, fs_random,
    fs_random_quick_restart]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 20`:
  - 13 full problem subsolvers: [default_lp, lb_tree_search, max_lp, no_lp,
    objective_lb_search, objective_shaving_search_max_lp,
    objective_shaving_search_no_lp, probing, probing_max_lp, pseudo_costs,
    quick_restart, quick_restart_no_lp, reduced_costs]
  - 5 first solution subsolvers: [fj_long_default, fj_short_default,
    fj_short_lin_default, fs_random, fs_random_quick_restart]
  - 13 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 32`:
  - 15 full problem subsolvers: [default_lp, lb_tree_search, max_lp, no_lp,
    objective_lb_search, objective_lb_search_max_lp, objective_lb_search_no_lp,
    objective_shaving_search_max_lp, objective_shaving_search_no_lp, probing,
    probing_max_lp, pseudo_costs, quick_restart, quick_restart_no_lp,
    reduced_costs]
  - 15 first solution subsolvers: [fj_long_default, fj_long_lin_default,
    fj_long_lin_random, fj_long_random, fj_short_default, fj_short_lin_default,
    fj_short_lin_random, fj_short_random, fs_random(2), fs_random_no_lp(2),
    fs_random_quick_restart(2), fs_random_quick_restart_no_lp]
  - 15 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls(3)]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]
- `solver.parameters.num_search_workers = 64`:
  - 15 full problem subsolvers: [default_lp, lb_tree_search, max_lp, no_lp,
    objective_lb_search, objective_lb_search_max_lp, objective_lb_search_no_lp,
    objective_shaving_search_max_lp, objective_shaving_search_no_lp, probing,
    probing_max_lp, pseudo_costs, quick_restart, quick_restart_no_lp,
    reduced_costs]
  - 47 first solution subsolvers: [fj_long_default(2), fj_long_lin_default(2),
    fj_long_lin_perturb(2), fj_long_lin_random(2), fj_long_perturb(2),
    fj_long_random(2), fj_short_default(2), fj_short_lin_default(2),
    fj_short_lin_perturb(2), fj_short_lin_random(2), fj_short_perturb(2),
    fj_short_random(2), fs_random(6), fs_random_no_lp(6),
    fs_random_quick_restart(6), fs_random_quick_restart_no_lp(5)]
  - 19 incomplete subsolvers: [feasibility_pump, graph_arc_lns, graph_cst_lns,
    graph_dec_lns, graph_var_lns, packing_precedences_lns,
    packing_rectangles_lns, packing_slice_lns, rins/rens, rnd_cst_lns,
    rnd_var_lns, scheduling_precedences_lns, violation_ls(7)]
  - 3 helper subsolvers: [neighborhood_helper, synchronization_agent,
    update_gap_integral]

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

You can specify the used full problem subsolvers manually by

```
# make sure list is empty
while solver.parameters.subsolvers:
   solver.parameters.subsolvers.pop()
# set new list
solver.parameters.subsolvers.extend(["default_lp", "fixed", "less_encoding", "no_lp", "max_lp", "pseudo_costs", "reduced_costs", "quick_restart", "quick_restart_no_lp", "lb_tree_search", "probing"])
```

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
solutions feasible. If you want to use more workers heuristically searching for
good solutions, you can specify `solver.parameters.min_num_lns_workers`. You can
also run only the incomplete subsolvers by setting
`solver.parameters.use_lns_only = True`, but this need to be combined with a
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

Quite often you want to find out what happens if you force some variables to a
specific value. Because you possibly want to do that multiple times, you do not
want to copy the whole model. CP-SAT has a nice option of adding assumptions
that you can clear afterwards, without needing to copy the object to test the
next assumptions. This is a feature also known from many SAT-solvers and CP-SAT
also only allows this feature for boolean literals. You cannot add any more
complicated expressions here, but for boolean literals it seems to be pretty
efficient. By adding some auxiliary boolean variables, you can also use this
technique to play around with more complicated expressions without the need to
copy the model. If you really need to temporarily add complex constraints, you
may have to copy the model using `model.CopyFrom` (maybe you also need to copy
the variables. Need to check that.).

```python
model.AddAssumptions([b1, b2.Not()])  # assume b1=True, b2=False
model.AddAssumption(b3)  # assume b3=True (single literal)
# ... solve again and analyse ...
model.ClearAssumptions()  # clear all assumptions
```

> An **assumption** is a temporary fixation of a boolean variable to true or
> false. It can be efficiently handled by a SAT-solver (and thus probably also
> by CP-SAT) and does not harm the learned clauses (but can reuse them).

### Hints

Maybe we already have a good intuition on how the solution will look like (this
could be, because we already solved a similar model, have a good heuristic,
etc.). In this case it may be useful, to tell CP-SAT about it, so it can
incorporate this knowledge into its search. For Mixed Integer Programming
Solver, this often yields a visible boost, even with mediocre heuristic
solutions. For CP-SAT I actually also encountered downgrades of the performance
if the hints were not great (but this may depend on the problem).

```python
model.AddHint(x, 1)  # Tell CP-SAT that x will probably be 1
model.AddHint(y, 2)  # and y probably be 2.
```

You can also find
[an official example](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/solution_hinting_sample_sat.py).

To make sure, your hints are actually correct, you can use the following
parameters to make CP-SAT throw an error if your hints are wrong.

```python
solver.parameters.debug_crash_on_bad_hint = True
```

If you have the feeling that your hints are not used, you may have made a
logical error in your model or just have a bug in your code. This parameter will
tell you about it.

(TODO: Have not tested this, yet)

### Logging

Sometimes it is useful to activate logging to see what is going on. This can be
achieved by setting the following two parameters.

```python
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
solver.log_callback = print  # (str)->None
```

If you get a doubled output, remove the last line.

The output can look as follows:

```
Starting CP-SAT solver v9.3.10497
Parameters: log_search_progress: true
Setting number of workers to 16

Initial optimization model '':
#Variables: 290 (#ints:1 in objective)
  - 290 in [0,17]
#kAllDiff: 34
#kLinMax: 1
#kLinear2: 2312 (#complex_domain: 2312)

Starting presolve at 0.00s
[ExtractEncodingFromLinear] #potential_supersets=0 #potential_subsets=0 #at_most_one_encodings=0 #exactly_one_encodings=0 #unique_terms=0 #multiple_terms=0 #literals=0 time=9.558e-06s
[Probing] deterministic_time: 0.053825 (limit: 1) wall_time: 0.0947566 (12427/12427)
[Probing]  - new integer bounds: 1
[Probing]  - new binary clause: 9282
[DetectDuplicateConstraints] #duplicates=0 time=0.00993671s
[DetectDominatedLinearConstraints] #relevant_constraints=2312 #work_done=14118 #num_inclusions=0 #num_redundant=0 time=0.0013379s
[DetectOverlappingColumns] #processed_columns=0 #work_done=0 #nz_reduction=0 time=0.00176239s
[ProcessSetPPC] #relevant_constraints=612 #num_inclusions=0 work=29376 time=0.0022503s
[Probing] deterministic_time: 0.0444515 (limit: 1) wall_time: 0.0820382 (11849/11849)
[Probing]  - new binary clause: 9282
[DetectDuplicateConstraints] #duplicates=0 time=0.00786558s
[DetectDominatedLinearConstraints] #relevant_constraints=2312 #work_done=14118 #num_inclusions=0 #num_redundant=0 time=0.000688681s
[DetectOverlappingColumns] #processed_columns=0 #work_done=0 #nz_reduction=0 time=0.000992311s
[ProcessSetPPC] #relevant_constraints=612 #num_inclusions=0 work=29376 time=0.00121334s

Presolve summary:
  - 0 affine relations were detected.
  - rule 'all_diff: expanded' was applied 34 times.
  - rule 'deductions: 10404 stored' was applied 1 time.
  - rule 'linear: simplified rhs' was applied 7514 times.
  - rule 'presolve: 0 unused variables removed.' was applied 1 time.
  - rule 'presolve: iteration' was applied 2 times.
  - rule 'variables: add encoding constraint' was applied 5202 times.

Presolved optimization model '':
#Variables: 5492 (#ints:1 in objective)
  - 5202 in [0,1]
  - 289 in [0,17]
  - 1 in [1,17]
#kAtMostOne: 612 (#literals: 9792)
#kLinMax: 1
#kLinear1: 10404 (#enforced: 10404)
#kLinear2: 2312 (#complex_domain: 2312)

Preloading model.
#Bound   0.45s best:inf   next:[1,17]     initial_domain

Starting Search at 0.47s with 16 workers.
9 full subsolvers: [default_lp, no_lp, max_lp, reduced_costs, pseudo_costs, quick_restart, quick_restart_no_lp, lb_tree_search, probing]
Interleaved subsolvers: [feasibility_pump, rnd_var_lns_default, rnd_cst_lns_default, graph_var_lns_default, graph_cst_lns_default, rins_lns_default, rens_lns_default]
#1       0.71s best:17    next:[1,16]     quick_restart_no_lp fixed_bools:0/11849
#2       0.72s best:16    next:[1,15]     quick_restart_no_lp fixed_bools:289/11849
#3       0.74s best:15    next:[1,14]     no_lp fixed_bools:867/11849
#Bound   1.30s best:15    next:[8,14]     max_lp initial_propagation
#Done    3.40s max_lp
#Done    3.40s probing

Sub-solver search statistics:
  'max_lp':
     LP statistics:
       final dimension: 2498 rows, 5781 columns, 106908 entries with magnitude in [6.155988e-02, 1.000000e+00]
       total number of simplex iterations: 3401
       num solves:
         - #OPTIMAL: 6
         - #DUAL_UNBOUNDED: 1
         - #DUAL_FEASIBLE: 1
       managed constraints: 5882
       merged constraints: 3510
       coefficient strenghtenings: 19
       num simplifications: 1
       total cuts added: 3534 (out of 4444 calls)
         - 'CG': 1134
         - 'IB': 150
         - 'MIR_1': 558
         - 'MIR_2': 647
         - 'MIR_3': 490
         - 'MIR_4': 37
         - 'MIR_5': 60
         - 'MIR_6': 20
         - 'ZERO_HALF': 438

  'reduced_costs':
     LP statistics:
       final dimension: 979 rows, 5781 columns, 6456 entries with magnitude in [3.333333e-01, 1.000000e+00]
       total number of simplex iterations: 1369
       num solves:
         - #OPTIMAL: 15
         - #DUAL_FEASIBLE: 51
       managed constraints: 2962
       merged constraints: 2819
       shortened constraints: 1693
       coefficient strenghtenings: 675
       num simplifications: 1698
       total cuts added: 614 (out of 833 calls)
         - 'CG': 7
         - 'IB': 439
         - 'LinMax': 1
         - 'MIR_1': 87
         - 'MIR_2': 80

  'pseudo_costs':
     LP statistics:
       final dimension: 929 rows, 5781 columns, 6580 entries with magnitude in [3.333333e-01, 1.000000e+00]
       total number of simplex iterations: 1174
       num solves:
         - #OPTIMAL: 14
         - #DUAL_FEASIBLE: 33
       managed constraints: 2923
       merged constraints: 2810
       shortened constraints: 1695
       coefficient strenghtenings: 675
       num simplifications: 1698
       total cuts added: 575 (out of 785 calls)
         - 'CG': 5
         - 'IB': 400
         - 'LinMax': 1
         - 'MIR_1': 87
         - 'MIR_2': 82

  'lb_tree_search':
     LP statistics:
       final dimension: 929 rows, 5781 columns, 6650 entries with magnitude in [3.333333e-01, 1.000000e+00]
       total number of simplex iterations: 1249
       num solves:
         - #OPTIMAL: 16
         - #DUAL_FEASIBLE: 14
       managed constraints: 2924
       merged constraints: 2809
       shortened constraints: 1692
       coefficient strenghtenings: 675
       num simplifications: 1698
       total cuts added: 576 (out of 785 calls)
         - 'CG': 8
         - 'IB': 400
         - 'LinMax': 2
         - 'MIR_1': 87
         - 'MIR_2': 79


Solutions found per subsolver:
  'no_lp': 1
  'quick_restart_no_lp': 2

Objective bounds found per subsolver:
  'initial_domain': 1
  'max_lp': 1

Improving variable bounds shared per subsolver:
  'no_lp': 579
  'quick_restart_no_lp': 1159

CpSolverResponse summary:
status: OPTIMAL
objective: 15
best_bound: 15
booleans: 12138
conflicts: 0
branches: 23947
propagations: 408058
integer_propagations: 317340
restarts: 23698
lp_iterations: 1174
walltime: 3.5908
usertime: 3.5908
deterministic_time: 6.71917
gap_integral: 11.2892
```

The log is actually very interesting to understand CP-SAT, but also to learn
about the optimization problem at hand. It gives you a lot of details on, e.g.,
how many variables could be directly removed or which techniques contributed to
lower and upper bounds the most.

As the log can be quite overwhelming, I developed a small tool to visualize and
comment the log. You can just copy and paste your log into it, and it will
automatically show you the most important details. Also be sure to check out the
examples in it.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cpsat-log-analyzer.streamlit.app/)
[![d-krupke - CP-SAT Log Analyzer](https://img.shields.io/badge/d--krupke-CP--SAT%20Log%20Analyzer-blue?style=for-the-badge&logo=github)](https://github.com/d-krupke/CP-SAT-Log-Analyzer)

|                                                                                                                            ![Search Progress](https://github.com/d-krupke/cpsat-primer/blob/main/images/search_progress.png)                                                                                                                            |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| A plot of the search progress over time as visualized by the log analyzer by utilizing the information from the log (a different log than displayed above). Such a plot helps you understand what part of your problem is more challenging: Finding a good solution or proving its quality. Based on that you can implement respective countermeasures. |

You can also find an older explanation of the log
[here](https://github.com/d-krupke/cpsat-primer/blob/main/understanding_the_log.md).

### Decision Strategy

In the end of this section, a more advanced parameter that looks interesting for
advanced users as it gives some insights into the search algorithm, **but is
probably better left alone**.

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

> **CAVEAT:** In the documentation there is a warning about the completeness of
> the domain strategy. I am not sure, if this is just for custom strategies or
> you have to be careful in general. So be warned.

```python
model.AddDecisionStrategy([x], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

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
