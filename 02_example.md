<!-- EDIT THIS PART VIA 02_example.md -->

<a name="02-example"></a>

## Example

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
x = model.new_int_var(0, 100, "x")
y = model.new_int_var(0, 100, "y")

# Constraints
model.add(x + y <= 30)

# Objective
model.maximize(30 * x + 50 * y)

# Solve
solver = cp_model.CpSolver()
status = solver.solve(model)

# The status tells us if we were able to compute an optimal solution.
assert status == cp_model.OPTIMAL, "Could not find optimal solution."

# Print the optimal solution.
print(f"x={solver.value(x)},  y={solver.value(y)}")
```

    x=0,  y=30

Pretty easy, right? For solving a generic problem, not just one specific
instance, you would of course create a dictionary or list of variables and use
something like `model.add(sum(vars)<=n)`, because you do not want to create the
model by hand for larger instances.

For larger models, CP-SAT will unfortunately not always able to compute an
optimal solution. However, the good news is that the solver will likely still
find a satisfactory solution and provide a bound on the optimal solution. Once
you reach this point, understanding how to interpret the solver's log becomes
crucial for analyzing the solver's performance. We will learn more about this
later.

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
>>> x = model.new_int_var(0, 100, "x")
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

If you are not yet satisfied,
[this folder contains many Jupyter Notebooks with examples from the developers](https://github.com/google/or-tools/tree/stable/examples/notebook/sat).
For example

- [multiple_knapsack_sat.ipynb](https://github.com/google/or-tools/blob/stable/examples/notebook/sat/multiple_knapsack_sat.ipynb)
  shows how to solve a multiple knapsack problem.
- [nurses_sat.ipynb](https://github.com/google/or-tools/blob/stable/examples/notebook/sat/nurses_sat.ipynb)
  shows how to schedule the shifts of nurses.
- [bin_packing_sat.ipynb](https://github.com/google/or-tools/blob/stable/examples/notebook/sat/bin_packing_sat.ipynb)
  shows how to solve a bin packing problem.
- ... (if you know more good examples I should mention here, please let me
  know!)

Now that you have seen a minimal model, let us explore the various options
available for problem modeling. While an experienced optimizer might be able to
handle most problems using just the elements previously discussed, clearly
expressing your intentions can help CP-SAT optimize your problem more
effectively.

> ![TIP]
>
> If you are transitioning from Mixed Integer Programming (MIP), you may be
> accustomed to manually implementing higher-level constraints to optimize your
> [Big-Ms](https://en.wikipedia.org/wiki/Big_M_method) for better performance,
> instead of relying on the modeling interface. With CP-SAT, these manual
> adjustments are generally unnecessary. CP-SAT relies less on linear relaxation
> compared to MIP solvers and can usually efficiently manage logical constraints
> thanks to its underlying SAT-solver. Dare to use the higher-level constraints!

---
