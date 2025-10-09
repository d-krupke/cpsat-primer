## MathOpt as a Modeling Layer

<a name="chapters-mathopt"></a>

Google OR-Tools recently introduced a new modeling layer called
[MathOpt](https://developers.google.com/optimization/math_opt). It provides a
solver-agnostic API for defining and solving mathematical optimization problems,
especially linear programs (LPs) and mixed-integer programs (MIPs). It is meant
as a simpler and more modern alternative to the older `pywraplp` interface, with
a stronger focus on usability and performance.

This is exciting because many optimization problems can naturally be formulated
as LPs or MIPs, and CP-SAT is not always the right tool for those. As discussed
earlier, working with continuous variables and coefficients in CP-SAT is not
very convenient, since they have to be discretized carefully. With MathOpt, you
can directly use continuous variables and floating-point coefficients when using
a solver that supports them (for example, HiGHS or Gurobi). For certain types of
problems, LP/MIP solvers can also be much more efficient than CP-SAT. When you
model your problem with MathOpt, you can easily switch between different solvers
and see which one works best for your case.

What is the catch when using MathOpt? It introduces a small amount of overhead
compared to calling a solver’s native API directly, and it currently supports
only linear constraints—that is, the ones you would add in CP-SAT using the
`add` method, excluding the `!=`, `<`, and `>` operators. MathOpt itself
supports continuous and unbounded variables as well as floating-point
coefficients, but these features are not available when you select CP-SAT as the
backend. If your model contains continuous variables, you will get an error,
since CP-SAT only supports integer variables and coefficients.

If you want to compare the performance of CP-SAT with a MIP solver, you still
need to discretize continuous variables and coefficients when your problem
requires it. Fortunately, many practical problems are purely integral, so you
might not even notice this limitation in practice.

> [!NOTE]
>
> Another modeling language aimed at the constraint programming community is
> [CPMpy](https://cpmpy.readthedocs.io/en/latest/index.html), which directly
> supports CP-SAT as a solver backend. In contrast, modeling tools commonly used
> in the mathematical optimization community—such as
> [Pyomo](https://pyomo.org/), [GAMS](https://www.gams.com/), and
> [AMPL](https://ampl.com/)—do not support CP-SAT.

In the following, I give a brief overview of the MathOpt API.

### Import and Model Creation

As in CP-SAT, you first need to import the MathOpt module and create a model
instance.

```python
from ortools.math_opt.python import mathopt as mo

model = mo.Model(name="optional_name")
```

### Variables

Next, you can start adding variables to the model. MathOpt supports continuous,
binary, and integer variables. By default, continuous and integer variables are
non-negative (i.e., they have a lower bound of 0). You can specify different
bounds using the `lb` and `ub` parameters.

```python
x = model.add_variable(lb=0.0, ub=10.0, name="x")  # Continuous variable in [0, 10]
# Unlike CP-SAT, you do not have to name variables
nameless_x = model.add_variable(lb=0.0, ub=10.0)  # Nameless variable
x_unbounded = model.add_variable(lb=-math.inf, ub=math.inf, name="x_unbounded")  # Unbounded continuous variable
```

To create binary or integer variables, use the corresponding methods:

```python
y = model.add_binary_variable(name="y")  # Binary variable in {0, 1}
z = model.add_integer_variable(lb=0, ub=100, name="z")  # Integer variable in [0, 100]
# Integer variable with no upper bound
z_unbounded = model.add_integer_variable(name="z_unbounded")
# or explicitly
z_unbounded = model.add_integer_variable(lb=0, ub=math.inf, name="z_unbounded")
```

> [!WARNING]
>
> The default lower bound is 0 for all variables, and the upper bound is
> infinity for continuous and integer variables. If you want a variable that can
> take negative values, you must explicitly set the lower bound to `-math.inf`
> or another negative number.

### Constraints

You can add linear constraints to the model using the `add_linear_constraint`
method. There are two ways to do this: the inequality/equality form and the
boxed form.

```python
# Inequality/equality form using Python operators
model.add_linear_constraint(2 * x + 3 * y <= 10, name="constraint1")
model.add_linear_constraint(x + y + z == 5, name="constraint2")
model.add_linear_constraint(z - 2 * x >= 0, name="constraint3")

# Boxed form
model.add_linear_constraint(lb=0, expr=2 * x + 3 * y, ub=10, name="constraint1_boxed")
model.add_linear_constraint(lb=5, expr=x + y + z, ub=5, name="constraint2_boxed")  # equality
model.add_linear_constraint(lb=0, expr=z - 2 * x, ub=math.inf, name="constraint3_boxed")  # inequality

# Constraints do not have to be named
model.add_linear_constraint(4 * x + y <= 20)
```

If you want to sum over a list of terms, you can use standard Python sums as in
CP-SAT. However, MathOpt provides a more efficient function, `mo.fast_sum`,
which is optimized for performance.

```python
terms = [i * x for i in range(1, 11)]  # Create a list of terms
model.add_linear_constraint(mo.fast_sum(terms) <= 100, name="sum_constraint")
```

You can also use expression handles, which can make the model more readable and
easier to maintain.

```python
in_vars = [mo.variable(name=f"in_{i}") for i in range(1, 6)]
out_vars = [mo.variable(name=f"out_{i}") for i in range(1, 6)]
incoming_flow = mo.fast_sum(in_vars)
outgoing_flow = mo.fast_sum(out_vars)
model.add_linear_constraint(incoming_flow == outgoing_flow, name="flow_conservation")
```

### Objective

You can define the model’s objective using the `set_objective` method and
specify whether you want to maximize or minimize it.

```python
model.set_objective(3 * x + 4 * y + 2 * z, is_maximize=True)  # Maximize
model.set_objective(x + 2 * z, is_maximize=False)  # Minimize
```

### Solving

To solve the model, use the `mo.solve` function, which takes the model, the
solver type, and optional solver parameters.

```python
params = mo.SolveParameters(
    time_limit=timedelta(seconds=30),  # Time limit of 30 seconds
    relative_gap_tolerance=0.01,  # 1% relative gap tolerance
    enable_output=True  # Enable solver output
)
result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)
```

MathOpt currently supports the following solvers: `GSCIP`, `GUROBI`, `GLOP`,
`CP_SAT`, `PDLP`, `GLPK`, `OSQP`, `ECOS`, `SCS`, `HIGHS`, and `SANTORINI`. If
you do not have a Gurobi license, I recommend using HiGHS or GSCIP, both of
which are open-source mixed-integer programming solvers.

### Inspecting Results

After solving the model, you can inspect the results through the `SolveResult`
object returned by `mo.solve`.

Check the termination reason:

```python
term = result.termination
print(f"Termination: {term.reason.name}")
if term.detail:
    print(f"Detail: {term.detail}")
```

Check whether a primal feasible solution was found:

```python
if result.has_primal_feasible_solution():
    print(f"Objective value: {result.objective_value()}")
    values = result.variable_values()
    print(f"x: {values.get(x)}, y: {values.get(y)}, z: {values.get(z)}")
else:
    print("No primal feasible solution found.")
```

The `variable_values()` method returns a mapping from variables to their values.
You can also pass a list of variables to get their values in the same order,
which can be convenient if you only need a subset.

```python
values = result.variable_values([x, y, z])
print(f"x: {values[0]}, y: {values[1]}, z: {values[2]}")
```

> [!NOTE]
>
> If you are solving an LP and the underlying solver supports dual values, you
> can also access the dual values of the constraints using the `dual_values()`
> method of the `SolveResult` object. Remember that in this case, you need to
> keep the handle to the constraints when you create them, e.g.,
> `constraint = model.add_linear_constraint(...)`. There are further features
> available, such as callbacks and lazy constraints, which we do not cover here.
> However, the
> [examples](https://github.com/google/or-tools/tree/stable/ortools/math_opt/samples/python)
> show some nice use cases.

### Examples

Let us look at two examples that demonstrate how to model and solve optimization
problems with MathOpt.

#### Simplified Stigler Diet Problem

The **Simplified Stigler Diet Problem** is a classical optimization problem in
which the goal is to select nonnegative servings of various foods to minimize
the total cost while satisfying minimum nutritional requirements for calories,
protein, and calcium. This model serves as a small, illustrative example of how
mathematical optimization can be applied to dietary planning. The units and
values used here are for demonstration purposes only and should not be
interpreted as nutritional facts.

```python
from ortools.math_opt.python import mathopt as mo
from datetime import timedelta
# --- Data -----------------------------------------------------------------
foods = ["Wheat Flour", "Milk", "Cabbage", "Beef"]

# Cost per serving (EUR)
cost = {
    "Wheat Flour": 0.36,
    "Milk": 0.23,
    "Cabbage": 0.10,
    "Beef": 1.20,
}

# Nutrients per serving (approximate / illustrative)
#               Calories  Protein(g)  Calcium(mg)
calories =     {"Wheat Flour": 364.0, "Milk": 150.0, "Cabbage": 25.0,  "Beef": 250.0}
protein =      {"Wheat Flour": 10.0,  "Milk": 8.0,   "Cabbage": 1.3,   "Beef": 26.0}
calcium =      {"Wheat Flour": 15.0,  "Milk": 285.0, "Cabbage": 40.0,  "Beef": 20.0}

# Minimum requirements
req = {
    "Calories": 2000.0,   # kcal
    "Protein": 55.0,      # g
    "Calcium": 800.0,     # mg
}

# --- Model ----------------------------------------------------------------
model = mo.Model(name="stigler_diet")

# Decision: servings of each food (continuous, ≥ 0)
# You could switch to integers to create a MIP variant.
servings = {f: model.add_variable(lb=0.0, name=f"servings[{f}]") for f in foods}

# Optional upper bounds to keep results tidy
for f in foods:
    model.add_linear_constraint(servings[f] <= 20.0, name=f"cap[{f}]")

# --- Nutrient constraints --------------------------------------------------
model.add_linear_constraint(
    mo.fast_sum(calories[f] * servings[f] for f in foods) >= req["Calories"],
    name="nutrients[Calories]",
)
model.add_linear_constraint(
    mo.fast_sum(protein[f] * servings[f] for f in foods) >= req["Protein"],
    name="nutrients[Protein]",
)
model.add_linear_constraint(
    mo.fast_sum(calcium[f] * servings[f] for f in foods) >= req["Calcium"],
    name="nutrients[Calcium]",
)

# --- Objective: minimize total cost ---------------------------------------
model.set_objective(
    mo.fast_sum(cost[f] * servings[f] for f in foods),
    is_maximize=False,
)

# --- Solve ----------------------------------------------------------------
params = mo.SolveParameters(
    time_limit=timedelta(seconds=10),
    relative_gap_tolerance=1e-6,  # LP, so we can be tight
    enable_output=False,
)

result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)

# --- Report ----------------------------------------------------------------
term = result.termination
print(f"Termination: {term.reason.name}")

if not result.has_primal_feasible_solution():
    print("No feasible solution found.")
    return

print(f"Minimum cost: €{result.objective_value():.2f}")

# Extract chosen servings (suppress near-zeros)
values = result.variable_values()
print("\nServings:")
for f in foods:
    qty = values.get(servings[f], 0.0)
    if abs(qty) > 1e-9:
        print(f"  {f:12s}: {qty:8.3f}")

# Compute realized totals
tot_cal = sum(calories[f] * values.get(servings[f], 0.0) for f in foods)
tot_pro = sum(protein[f]  * values.get(servings[f], 0.0) for f in foods)
tot_calcium = sum(calcium[f] * values.get(servings[f], 0.0) for f in foods)

print("\nNutrient totals (minimum required in parentheses):")
print(f"  Calories: {tot_cal:.1f}  ({req['Calories']})")
print(f"  Protein : {tot_pro:.1f}  ({req['Protein']})")
print(f"  Calcium : {tot_calcium:.1f}  ({req['Calcium']})")
```

#### Set Cover

The **Set Cover Problem** is a classic combinatorial optimization problem. Given
a universe $U$ of elements and a collection of subsets $S \subseteq 2^U$, each
with an associated integer cost, the goal is to select a minimum-cost collection
of subsets such that every element in $U$ is covered by at least one selected
subset.

This problem appears in resource allocation, scheduling, and network design. A
standard integer formulation is:

- **Variables:** For each subset $S$, a binary variable $z[S] \in \{0, 1\}$
  indicates whether subset $S$ is chosen.
- **Constraints:** For each element $u \in U$, the sum of $z[S]$ over all
  subsets $S$ containing $u$ must be at least 1.
- **Objective:** Minimize the total cost $\sum_{S} \text{cost}[S] \cdot z[S]$.

Since all variables are integral, this problem can be modeled using CP-SAT and
compared directly with MIP solvers such as Gurobi or HiGHS.

```python
from ortools.math_opt.python import mathopt as mo
from datetime import timedelta

U = {1, 2, 3, 4, 5, 6}
subsets = {
    "S1": {1, 2, 3},
    "S2": {2, 4},
    "S3": {3, 5, 6},
    "S4": {1, 4, 6},
    "S5": {2, 5},
    "S6": {4, 5, 6},
}
cost = {"S1": 4, "S2": 2, "S3": 3, "S4": 3, "S5": 2, "S6": 4}

model = mo.Model(name="set_cover_cp_sat")

# Decision variables
z = {s: model.add_binary_variable(name=f"pick[{s}]") for s in subsets}

# Cover constraints
for u in U:
    model.add_linear_constraint(
        mo.fast_sum(z[s] for s in subsets if u in subsets[s]) >= 1,
        name=f"cover[{u}]",
    )

# Objective
model.set_objective(
    mo.fast_sum(cost[s] * z[s] for s in subsets),
    is_maximize=False,
)

# Solve with CP-SAT
params = mo.SolveParameters(time_limit=timedelta(seconds=10), enable_output=True)
result = mo.solve(model, solver_type=mo.SolverType.CP_SAT, params=params)

print(f"[SetCover] Termination: {result.termination.reason.name}")
if not result.has_primal_feasible_solution():
    print("[SetCover] No feasible solution found.")
    return

vals = result.variable_values()
chosen = [s for s in subsets if int(round(vals.get(z[s], 0.0)))]
total_cost = sum(cost[s] for s in chosen)
print(f"[SetCover] Minimum total cost: {total_cost}")
print("[SetCover] Chosen subsets:", ", ".join(chosen))
```
