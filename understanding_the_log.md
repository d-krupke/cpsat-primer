# Understanding the CP-SAT log

> As the log can be quite overwhelming, I developed a small tool to visualize
> and comment the log. You can just copy and paste your log into it, and it will
> automatically show you the most important details. Also be sure to check out
> the examples in it.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cpsat-log-analyzer.streamlit.app/)
[![d-krupke - CP-SAT Log Analyzer](https://img.shields.io/badge/d--krupke-CP--SAT%20Log%20Analyzer-blue?style=for-the-badge&logo=github)](https://github.com/d-krupke/CP-SAT-Log-Analyzer)

The text below is partially deprecated and supposed to be completely replaced by
the tool, which will have extensive documentation.

> **WORK IN PROGRESS**

Just printing version and parameters:

```
Starting CP-SAT solver v9.3.10497
Parameters: log_search_progress: true
Setting number of workers to 16
```

Description of the model how you created it. For example we used 34
`AllDifferent`, 1 `MaxEquality`, and 2312 linear constraints with 2 variables.

```
Initial optimization model '':
#Variables: 290 (#ints:1 in objective)
  - 290 in [0,17]
#kAllDiff: 34
#kLinMax: 1
#kLinear2: 2312 (#complex_domain: 2312)
```

## Presolve

We first see multiple rounds of domain reduction, expansion, equivalence
checking, substitution, and probing.

```
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
```

Here we see for example that the `AllDifferent` constraint was expanded 34
times.

```
Presolve summary:
  - 0 affine relations were detected.
  - rule 'all_diff: expanded' was applied 34 times.
  - rule 'deductions: 10404 stored' was applied 1 time.
  - rule 'linear: simplified rhs' was applied 7514 times.
  - rule 'presolve: 0 unused variables removed.' was applied 1 time.
  - rule 'presolve: iteration' was applied 2 times.
  - rule 'variables: add encoding constraint' was applied 5202 times.

```

Here we see the optimization model with its variables and constraints. For
example, we have 5492 variables with 5202 of them being boolean, 289 of them
being in {0, 1, 2, ... , 17} and 1 of them being in {1, 2, ..., 17}. Afterwards
come the different constraints that are used internally.

```
Presolved optimization model '':
#Variables: 5492 (#ints:1 in objective)
  - 5202 in [0,1]
  - 289 in [0,17]
  - 1 in [1,17]
#kAtMostOne: 612 (#literals: 9792)
#kLinMax: 1
#kLinear1: 10404 (#enforced: 10404)
#kLinear2: 2312 (#complex_domain: 2312)
```

## Search progress

```
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
```

The list has the following columns:

1. Event or solution count (e.g. `#1` or `#Done`)
2. Time in seconds (e.g. `0.71s`)
3. Best objective (e.g. `best:17` or `best:inf`)
4. Next objective range (e.g. `next:[8,14]` for looking for better solutions
   with an objective between 8 and 14.) The first number is the current lower
   bound.
5. Info on how the solution/event was achieved.

## Subsolver statistics

### Statistics on the internal LPs

You will notice that the LP solver returns only three different states:

1. OPTIMAL: An optimal linear relaxation was found.
2. DUAL_UNBOUNDED: The linear relaxation is infeasible (because the dual problem
   is unbounded).
3. DUAL_FEASIBLE: We have a feasible solution for the dual problem (which gives
   us at least a lower bound).

Note that the dual simplex algorithm will only give us a solution for the linear
relaxation if it is optimal.

We can see that the LP in `max_lp` has much more constraints (2498 rows) than
the other solvers. On top come the cutting planes (cuts) of various types that
can be deduced from the LP to improve the integrality. The numbers of simplex
iterations are surprisingly low: Usually simplex can need quite a lot of
iterations.

```
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
```

Which solvers actually found solutions? In this case, both solvers do not make
use of linear programming.

```
Solutions found per subsolver:
  'no_lp': 1
  'quick_restart_no_lp': 2
```

Which solvers were able to improve the lower bounds? While the LP-based solver
were not able to provide good solutions the solver that makes maximal usage of
linear programming was able to proof the lower bound.

```
Objective bounds found per subsolver:
  'initial_domain': 1
  'max_lp': 1
```

The bounds on individual variables one the other hand were best improved by the
solvers without linear programming.

```
Improving variable bounds shared per subsolver:
  'no_lp': 579
  'quick_restart_no_lp': 1159
```

## Summary

The final `CpSolverResponse`, which is defined and partially commented
[here](https://github.com/google/or-tools/blob/49b6301e1e1e231d654d79b6032e79809868a70e/ortools/sat/cp_model.proto#L704).

```bash
CpSolverResponse summary:
status: OPTIMAL  # We solved the problem to optimality
objective: 15  # We found a solution with value 15
best_bound: 15  # The proofed lower bound is 15
booleans: 12138
conflicts: 0
branches: 23947
propagations: 408058  # propagation of boolean variables
integer_propagations: 317340  # propagation of integer variables
restarts: 23698
lp_iterations: 1174
walltime: 3.5908  # runtime in seconds
usertime: 3.5908
deterministic_time: 6.71917
gap_integral: 11.2892  # "The integral of log(1 + absolute_objective_gap) over time."
```
