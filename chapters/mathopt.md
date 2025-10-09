## MathOpt as a Modeling Layer

<a name="chapters-mathopt"></a>

Google OR-Tools recently added a new modeling layer called
[MathOpt](https://developers.google.com/optimization/math_opt), which provides a
solver-agnostics API for defining and solving mathematical optimization
problems, especially linear programs (LPs) and mixed-integer programs (MIPs). It
is designed to be a simpler and more modern alternative to the older `pywraplp`
interface, with a focus on usability and performance. Why is this exciting?
Because many optimization problems can be expressed as LPs or MIPs, and CP-SAT
is not always the best tool for the job. First, as we already know, having
continuous variables and coefficients is not very convenient in CP-SAT, since we
carefully need to discretize them. With MathOpt, you can directly use continuous
variables as well as floating point coefficients, when using a solver that
supports them (e.g., HiGHS or Gurobi). Second, for certain problem types, the
techniques used in LP/MIP solvers can be much more efficient than CP-SAT's. When
modelling the problem with MathOpt, you can easily switch between different
solvers and see which one works best for your specific problem.

What is the catch of using MathOpt? It creates some overhead compared to using a
solver's native API directly and it currently also only supports linear
constraints, i.e., those you add in CP-SAT via the `add` method, excluding the
`!=`, `<`, `>` operators. Also, while MathOpt supports continuous and unbounded
variables as well as floating point coefficients, you will get an error if you
try to use them with CP-SAT, since CP-SAT does not support them. If you want to
compare the performance of CP-SAT with a MIP solver, you will therefore still
need to discretize continuous variables and coefficients (if your problem
requires it). Luckily, many problems in practice are purely integral, thus you
may would not even have noticed it if I wouldn't have mentioned it.

> [!NOTE]
>
> Another modelling language that is directed at the constraint programming
> instead of the mathematical optimization community is
> [CPMpy](https://cpmpy.readthedocs.io/en/latest/index.html). In the
> mathematical optimization community, the use of modelling tools such as
> [Pyomo](https://pyomo.org/), [GAMS](https://www.gams.com/), or
> [AMPL](https://ampl.com/), are quite common. However, these do not support
> CP-SAT as a solver backend.

In the following, I will give a brief overview of the MathOpt API.

### Import and Model Creation

Same as in CP-SAT, you first need to import the MathOpt module and create a
model instance.

```python
from ortools.math_opt.python import mathopt as mo

model = mo.Model(name="optional_name")
```

### Variables

Next, we can start adding variables to the model. We can add continuous, binary,
and integer variables. By default, continuous and integer variables are
non-negative (i.e., have a lower bound of 0). You can specify different bounds
using the `lb` and `ub` parameters.

```python
x = model.add_variable(lb=0.0, ub=10.0, name="x")  # Continuous variable in [0, 10]
# contrary to CP-SAT, you don't have to name variables
nameless_x = model.add_variable(lb=0.0, ub=10.0)  # Nameless variable
x_unbounded = model.add_variable(lb=-math.inf, ub=math.inf, name="x_unbounded")  # Unbounded continuous variable
```

If you want to have a binary or integer variable, you can use the corresponding
methods:

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
> The default lower bound is 0 for all variables, and infinity for continuous
> and integer variables. If you want to have a variable that can take negative
> values, you need to explicitly set the lower bound to `-math.inf` or any other
> negative value.

### Constraints

You can add linear constraints to the model using the `add_linear_constraint`
method. There are two ways to do this: using the inequality/equality form or the
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

If you want to sum over a list, you can use sums, just like in CP-SAT. However,
MathOpt provides a more efficient way to do this using `mo.fast_sum`, which is
optimized for performance.

```python
terms = [i * x for i in range(1, 11)]  # Create a list of terms
model.add_linear_constraint(mo.fast_sum(terms) <= 100, name="sum_constraint")
```

You can also use expression handles, which can be easier to read and maintain.

```python
in_vars = [mo.variable(name=f"in_{i}") for i in range(1, 6)]
out_vars = [mo.variable(name=f"out_{i}") for i in range(1, 6)]
incoming_flow = mo.fast_sum(in_vars)
outgoing_flow = mo.fast_sum(out_vars)
model.add_linear_constraint(incoming_flow == outgoing_flow, name="flow_conservation")
```

### Objective

You can set the objective of the model using the `set_objective` method. You can
specify whether you want to maximize or minimize the objective.

```python
model.set_objective(3 * x + 4 * y + 2 * z, is_maximize=True)  # Maximize
model.set_objective(x + 2 * z, is_maximize=False)  # Minimize
```

### Solving

To solve the model, you can use the `mo.solve` function, which takes the model,
the solver type, and optional solver parameters.

```python
params = mo.SolveParameters(
    time_limit=timedelta(seconds=30),  # Time limit of 30 seconds
    relative_gap_tolerance=0.01,  # 1% relative gap tolerance
    enable_output=True  # Enable solver output
)
result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)
```

At this point, MathOpt supports the following solvers: `GSCIP`, `GUROBI`,
`GLOP`, `CP_SAT`, `PDLP`, `GLPK`, `OSQP`, `ECOS`, `SCS`, `HIGHS`, `SANTORINI`.
If you do not have a license for Gurobi, I recommend using HiGHS or GSCIP which
are open-source Mixed Integer Programming solvers.

### Inspecting Results

After solving the model, you can inspect the results using the `SolveResult`
object returned by `mo.solve`.

Check the termination reason:

```python
term = result.termination
    print(f"Termination: {term.reason.name}")
    if term.detail:
        print(f"Detail: {term.detail}")
```

Check if a primal feasible solution was found:

```python
if result.has_primal_feasible_solution():
    print(f"Objective value: {result.objective_value()}")
    values = result.variable_values()
    print(f"x: {values.get(x)}, y: {values.get(y)}, z: {values.get(z)}")
else:
    print("No primal feasible solution found.")
```

`variable_values()` will return a mapping from variables to their values. You
can also pass a list of variables to get their values in the same order. This
can be useful if you only want to extract a subset of variables.

```python
values = result.variable_values([x, y, z])
print(f"x: {values[0]}, y: {values[1]}, z: {values[2]}")
```

### Examples

Let us see two examples of using MathOpt to model and solve optimization
problems.

### Simplified Stigler Diet Problem

The Simplified Stigler Diet problem is a classic optimization challenge where
the goal is to select nonnegative servings of various foods to minimize total
cost while meeting minimum nutritional requirements for calories, protein, and
calcium. This model serves as an illustrative example of how mathematical
optimization can be applied to dietary planning. Please note that the units and
values provided are for demonstration purposes only and should not be considered
as nutritional truths.

```python
from ortools.math_opt.python import mathopt as mo

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

# Requirements (minimums)
req = {
    "Calories": 2000.0,   # kcal
    "Protein": 55.0,      # grams
    "Calcium": 800.0,     # mg
}

# --- Model ----------------------------------------------------------------
model = mo.Model(name="stigler_diet")

# Decision: servings of each food (continuous, ≥ 0). You could switch to integers to show a MIP.
servings = {f: model.add_variable(lb=0.0, name=f"servings[{f}]") for f in foods}

# Optionally cap servings to keep things tidy (purely illustrative)
for f in foods:
    model.add_linear_constraint(servings[f] <= 20.0, name=f"cap[{f}]")

# Nutrient constraints ------------------------------------------------------
# Calories
model.add_linear_constraint(
    mo.fast_sum(calories[f] * servings[f] for f in foods) >= req["Calories"],
    name="nutrients[Calories]"
)
# Protein
model.add_linear_constraint(
    mo.fast_sum(protein[f] * servings[f] for f in foods) >= req["Protein"],
    name="nutrients[Protein]"
)
# Calcium
model.add_linear_constraint(
    mo.fast_sum(calcium[f] * servings[f] for f in foods) >= req["Calcium"],
    name="nutrients[Calcium]"
)

# Objective: minimize total cost -------------------------------------------
model.set_objective(
    mo.fast_sum(cost[f] * servings[f] for f in foods),
    is_maximize=False
)

# Solve parameters ----------------------------------------------------------
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

print(f"Min cost: €{result.objective_value():.2f}")

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

print("\nNutrient totals (min required in parentheses):")
print(f"  Calories: {tot_cal:.1f}  ({req['Calories']})")
print(f"  Protein : {tot_pro:.1f}  ({req['Protein']})")
print(f"  Calcium : {tot_calcium:.1f}  ({req['Calcium']})")
```

#### Set Cover

The Set Cover problem is a classic combinatorial optimization problem. Given a
universe $U$ of elements and a collection of subsets $S$ of $U$, each with an
associated integer cost, the goal is to select a minimum-cost collection of
subsets such that every element in $U$ is covered by at least one chosen subset.
This problem is widely studied in computer science and operations research due
to its applications in resource allocation, scheduling, and network design.

Below is a mathematical formulation of the integral Set Cover problem:

- **Variables:** For each subset $S$, introduce a binary variable
  $z[S] \in \{0, 1\}$ indicating whether subset $S$ is selected.
- **Constraints:** For each element $u \in U$, ensure that the sum of $z[S]$
  over all subsets $S$ containing $u$ is at least 1, i.e., every element is
  covered.
- **Objective:** Minimize the total cost, $\sum_S \text{cost}[S] \cdot z[S]$.

Since all variables are integral, this problem can be efficiently modeled and
solved using CP-SAT, and also compared with MIP solvers such as Gurobi or HiGHS.

```python
from ortools.math_opt.python import mathopt as mo

U = {1, 2, 3, 4, 5, 6}
subsets = {
    "S1": {1, 2, 3},
    "S2": {2, 4},
    "S3": {3, 5, 6},
    "S4": {1, 4, 6},
    "S5": {2, 5},
    "S6": {4, 5, 6},
}
cost = {"S1": 4, "S2": 2, "S3": 3, "S4": 3, "S5": 2, "S6": 4}  # integers

model = mo.Model(name="set_cover_cp_sat")

# Decision vars
z = {s: model.add_binary_variable(name=f"pick[{s}]") for s in subsets}

# Cover constraints
for u in U:
    model.add_linear_constraint(
        mo.fast_sum(z[s] for s in subsets if u in subsets[s]) >= 1,
        name=f"cover[{u}]"
    )

# Objective
model.set_objective(
    mo.fast_sum(cost[s] * z[s] for s in subsets),
    is_maximize=False
)

# Solve with CP-SAT
params = mo.SolveParameters(time_limit=timedelta(seconds=10), enable_output=True)
result = mo.solve(model, solver_type=mo.SolverType.CP_SAT, params=params)

print(f"[SetCover] Termination: {result.termination.reason.name}")
if not result.has_primal_feasible_solution():
    print("[SetCover] No feasible solution.")
    return

vals = result.variable_values()
chosen = [s for s in subsets if int(round(vals.get(z[s], 0.0)))]
total_cost = sum(cost[s] for s in chosen)
print(f"[SetCover] Min total cost: {total_cost}")
print("[SetCover] Chosen subsets:", ", ".join(chosen))
```
