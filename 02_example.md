<!-- EDIT THIS PART VIA 02_example.md -->

## Example
<a name="02-example"></a>

Before we dive into any internals, let us take a quick look at a simple
application of CP-SAT. This example is so simple that you could solve it by
hand, but know that CP-SAT would (probably) be fine with you adding a thousand
(maybe even ten- or hundred-thousand) variables and constraints more. The basic
idea of using CP-SAT is, analogous to MIPs, to define an optimization problem in
terms of variables, constraints, and objective function, and then let the solver
find a solution for it. We call such a formulation that can be understood by the
corresponding solver a _model_ for the problem. For people not familiar with
this
[declarative approach](https://programiz.pro/resources/imperative-vs-declarative-programming/),
you can compare it to SQL, where you also just state what data you want, not how
to get it. However, it is not purely declarative, because it can still make a
huge(!) difference how you model the problem and getting that right takes some
experience and understanding of the internals. You can still get lucky for
smaller problems (let us say a few hundred to thousands of variables) and obtain
optimal solutions without having an idea of what is going on. The solvers can
handle more and more 'bad' problem models effectively with every year.

> **Definition:** A _model_ in mathematical programming refers to a mathematical
> description of a problem, consisting of variables, constraints, and optionally
> an objective function that can be understood by the corresponding solver
> class. _Modelling_ refers to transforming a problem (instance) into the
> corresponding framework, e.g., by making all constraints linear as required
> for Mixed Integer Linear Programming. Be aware that the
> [SAT](https://en.wikipedia.org/wiki/SAT_solver)-community uses the term
> _model_ to refer to a (feasible) variable assignment, i.e., solution of a
> SAT-formula. If you struggle with this terminology, maybe you want to read
> this short guide on
> [Math Programming Modelling Basics](https://www.gurobi.com/resources/math-programming-modeling-basics/).

Our first problem has no deeper meaning, except of showing the basic workflow of
creating the variables (x and y), adding the constraint $x+y<=30$ on them,
setting the objective function (maximize $30*x + 50*y$), and obtaining a
solution:

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Variables
x = model.NewIntVar(0, 100, "x")  # you always need to specify an upper bound.
y = model.NewIntVar(0, 100, "y")
# there are also no continuous variables: You have to decide for a resolution and then work on integers.

# Constraints
model.Add(x + y <= 30)

# Objective
model.Maximize(30 * x + 50 * y)

# Solve
solver = cp_model.CpSolver()  # Contrary to Gurobi, model and solver are separated.
status = solver.Solve(model)
assert (
    status == cp_model.OPTIMAL
)  # The status tells us if we were able to compute a solution.
print(f"x={solver.Value(x)},  y={solver.Value(y)}")
print("=====Stats:======")
print(solver.SolutionInfo())
print(solver.ResponseStats())
```

    x=0,  y=30
    =====Stats:======
    default_lp
    CpSolverResponse summary:
    status: OPTIMAL
    objective: 1500
    best_bound: 1500
    booleans: 1
    conflicts: 0
    branches: 1
    propagations: 0
    integer_propagations: 2
    restarts: 1
    lp_iterations: 0
    walltime: 0.00289923
    usertime: 0.00289951
    deterministic_time: 8e-08
    gap_integral: 5.11888e-07

Pretty easy, right? For solving a generic problem, not just one specific
instance, you would of course create a dictionary or list of variables and use
something like `model.Add(sum(vars)<=n)`, because you do not want to create the
model by hand for larger instances.

The output you get may differ from the one above, because CP-SAT actually uses a
set of different strategies in parallel, and just returns the best one (which
can differ slightly between multiple runs due to additional randomness). This is
called a portfolio strategy and is a common technique in combinatorial
optimization, if you cannot predict which strategy will perform best.

In a later section, we will explore how to review and interpret the CP-SAT
solver's log. Analyzing the log is vital, especially when the solver's
performance is suboptimal. An expert can often pinpoint the cause of any issues
within seconds by examining the log, which is key to optimizing solver
efficiency.

### Mathematical Model

The mathematical model of the code above would usually be written by experts
something like this:

```math
\max 30x + 50y
```

```math
\text{s.t. } x+y \leq 30
```

```math
\quad 0\leq x \leq 100
```

```math
\quad 0\leq y \leq 100
```

```math
x,y \in \mathbb{Z}
```

The `s.t.` stands for `subject to`, sometimes also read as `such that`.

### Overloading

One aspect of using CP-SAT solver that often poses challenges for learners is
understanding operator overloading in Python and the distinction between the two
types of variables involved. In this context, `x` and `y` serve as mathematical
variables. That is, they are placeholders that will only be assigned specific
values during the solving phase. To illustrate this more clearly, let us explore
an example within the Python shell:

```pycon
>>> model = cp_model.CpModel()
>>> x = model.NewIntVar(0, 100, "x")
>>> x
x(0..100)
>>> type(x)
<class 'ortools.sat.python.cp_model.IntVar'>
>>> x + 1
sum(x(0..100), 1)
>>> x + 1 <= 1
<ortools.sat.python.cp_model.BoundedLinearExpression object at 0x7d8d5a765df0>
```

In this example, `x` is not a conventional number but a placeholder defined to
potentially assume any value between 0 and 100. When 1 is added to `x`, the
result is a new placeholder representing the sum of `x` and 1. Similarly,
comparing this sum to 1 produces another placeholder, which encapsulates the
comparison of the sum with 1. These placeholders do not hold concrete values at
this stage but are essential for defining constraints within the model.
Attempting operations like `if x + 1 <= 1: print("True")` will trigger a
`NotImplementedError`, as the condition `x+1<=1` cannot be evaluated directly.

Although this approach to defining models might initially seem perplexing, it
facilitates a closer alignment with mathematical notation, which in turn can
make it easier to identify and correct errors in the modeling process.

### More examples

Here are some further examples, if you are not yet satisfied:

- [N-queens](https://developers.google.com/optimization/cp/queens) (this one
  also gives you a quick introduction to constraint programming, but it may be
  misleading because CP-SAT is no classical
  [FD(Finite Domain)-solver](http://www.gameaipro.com/GameAIPro2/GameAIPro2_Chapter26_Rolling_Your_Own_Finite-Domain_Constraint_Solver.pdf).
  This example probably has been modified from the previous generation, which is
  also explained at the end.)
- [Employee Scheduling](https://developers.google.com/optimization/scheduling/employee_scheduling)
- [Job Shop Problem](https://developers.google.com/optimization/scheduling/job_shop)
- More examples can be found in
  [the official repository](https://github.com/google/or-tools/tree/stable/ortools/sat/samples)
  for multiple languages (yes, CP-SAT does support more than just Python). As
  the Python-examples are named in
  [snake-case](https://en.wikipedia.org/wiki/Snake_case), they are at the end of
  the list.

Ok. Now that you have seen a minimal model, let us look on what options we have
to model a problem. Note that an experienced optimizer may be able to model most
problems with just the elements shown above, but showing your intentions may
help CP-SAT optimize your problem better. Contrary to Mixed Integer Programming,
you also do not need to fine-tune any
[Big-Ms](https://en.wikipedia.org/wiki/Big_M_method) (a reason to model
higher-level constraints, such as conditional constraints that are only enforced
if some variable is set to true, in MIPs yourself, because the computer is not
very good at that).

---