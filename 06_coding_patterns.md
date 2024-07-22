<!-- EDIT THIS PART VIA 06_coding_patterns.md -->

<a name="06-coding-patterns"></a>

## Coding Patterns for Optimization Problems

<!-- START_SKIP_FOR_README -->

![Cover Image Patterns](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_4.webp)

<!-- STOP_SKIP_FOR_README -->

> [!WARNING]
>
> CP-SAT 9.9 recently changed its API to be more consistent with the commonly
> used Python style. Instead of `NewIntVar`, you can now also use `new_int_var`.
> The following part of the primer still uses the old style and will be updated
> soon.

In this section, we will explore various coding patterns that are essential for
structuring implementations for optimization problems using CP-SAT. While we
will not delve into the modeling of specific problems, our focus will be on
demonstrating how to organize your code to enhance its readability and
maintainability. These practices are crucial for developing robust and scalable
optimization solutions that can be easily understood, modified, and extended by
other developers. We will concentrate on basic patterns, as more complex
patterns are better understood within the context of larger problems and are
beyond the scope of this primer.

> [!WARNING]
>
> The naming conventions for patterns in optimization problems are not
> standardized. There is no comprehensive guide on coding patterns for
> optimization issues, and my insights are primarily based on personal
> experience. Most online examples tend to focus solely on the model, often
> presented as Jupyter notebooks or sequential scripts. The
> [gurobi-optimods](https://github.com/Gurobi/gurobi-optimods) provide the
> closest examples to production-ready code that I am aware of, yet they offer
> limited guidance on code structuring. I aim to address this gap, which many
> find challenging, though it is important to note that my approach is **highly
> opinionated**.

### Simple Function

For straightforward optimization problems, encapsulating the model creation and
solving within a single function is a practical approach. This method is best
suited for simpler cases due to its straightforward nature but lacks flexibility
for more complex scenarios. Parameters such as the time limit and optimality
tolerance can be customized via keyword arguments with default values.

The following Python function demonstrates solving a simple knapsack problem
using CP-SAT. To recap, in the knapsack problem, we select items - each with a
specific weight and value - to maximize total value without exceeding a
predefined weight limit. Given its simplicity, involving only one constraint,
the knapsack problem serves as an ideal model for introductory examples.

```python
from ortools.sat.python import cp_model
from typing import List


def solve_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
    *,
    time_limit: int = 900,
    opt_tol: float = 0.01,
) -> List[int]:
    # initialize the model
    model = cp_model.CpModel()
    n = len(weights)  # Number of items
    # Decision variables for items
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    # Capacity constraint
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    # Objective function to maximize value
    model.Maximize(sum(values[i] * x[i] for i in range(n)))
    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit  # Solver time limit
    solver.parameters.relative_gap_limit = opt_tol  # Solver optimality tolerance
    status = solver.Solve(model)
    # Extract solution
    return (
        # Return indices of selected items
        [i for i in range(n) if solver.Value(x[i])]
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else []
    )
```

### Logging the Model Building

When working with more complex optimization problems, logging the model-building
process can be essential to find and fix issues. Often, the problem lies not
within the solver but in the model itself.

In the following example, we add some basic logging to the solver function to
give us some insights into the model-building process. This logging can be
easily activated or deactivated by the logging framework, allowing us to use it
not only during development but also in production.

If you do not know about the logging framework of Python, this is an excellent
moment to learn about it. I consider it an essential skill for production code
and this and similar frameworks are used for most production code in any
language. The official Python documentation contains a
[good tutorial](https://docs.python.org/3/howto/logging.html).

```python
import logging
from ortools.sat.python import cp_model
from typing import List

_logger = logging.getLogger(__name__)  # get a logger for the current module


def solve_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
    *,
    time_limit: int = 900,
    opt_tol: float = 0.01,
) -> List[int]:
    _logger.debug("Building the knapsack model")
    # initialize the model
    model = cp_model.CpModel()
    n = len(weights)  # Number of items
    _logger.debug("Number of items: %d", n)
    if n > 0:
        _logger.debug(
            "Min/Mean/Max weight: %d/%.2f/%d",
            min(weights),
            sum(weights) / n,
            max(weights),
        )
        _logger.debug(
            "Min/Mean/Max value: %d/%.2f/%d", min(values), sum(values) / n, max(values)
        )
    # Decision variables for items
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    # Capacity constraint
    model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    # Objective function to maximize value
    model.Maximize(sum(values[i] * x[i] for i in range(n)))
    # Log the model
    _logger.debug("Model created with %d items", n)
    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit  # Solver time limit
    solver.parameters.relative_gap_limit = opt_tol  # Solver optimality tolerance
    _logger.debug(
        "Starting the solution process with time limit %d seconds", time_limit
    )
    status = solver.Solve(model)
    # Extract solution
    selected_items = (
        [i for i in range(n) if solver.Value(x[i])]
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else []
    )
    _logger.debug("Selected items: %s", selected_items)
    return selected_items
```

We will not use logging in the following examples to save space, but you should
consider adding it to your code.

> [!TIP]
>
> A great hack you can do with the logging framework is that you can easily hook
> into your code and do analysis beyond the simple logging. You can simply write
> a handler that, e.g., waits for the tag `"Selected items: %s"` and then can
> directly access the selected items, as the actual object is passed to the
> handler (and not just the string). This can be very useful to gather
> statistics or to visualize the search process.

### Custom Data Classes for Instances, Configurations, and Solutions

Incorporating serializable data classes to manage instances, configurations, and
solutions significantly enhances the readability and maintainability of your
code. These classes also facilitate the documentation process, testing, and
ensure data consistency across larger projects where data exchange among
different components is necessary.

**Implemented Changes:** We introduce data classes using
[Pydantic](https://docs.pydantic.dev/latest/), a popular Python library that
supports data validation and settings management through Python type
annotations. The changes include:

- **Instance Class**: Defines the knapsack problem with attributes for weights,
  values, and capacity. It includes a validation method to ensure that the
  number of weights matches the number of values, thereby enhancing data
  integrity.
- **Configuration Class**: Manages solver settings such as time limits and
  optimality tolerance, allowing for easy adjustments and fine-tuning of the
  solver’s performance. Default values ensure backward compatibility,
  facilitating the seamless integration of older configurations with new
  parameters.
- **Solution Class**: Captures the outcome of the optimization process,
  including which items were selected, the objective value, and the upper bound
  of the solution. This class allows us to add additional information, instead
  of just returning the pure solution. It also allows us to later extend the
  attached information without breaking the API, by just making the new entries
  optional or providing a default value. For example, you may be interested in
  the solution time that was required to find the solution.

```python
from ortools.sat.python import cp_model
from pydantic import BaseModel, PositiveInt, List, NonNegativeFloat


class KnapsackInstance(BaseModel):
    weights: List[PositiveInt]  # the weight of each item
    values: List[PositiveInt]  # the value of each item
    capacity: PositiveInt  # the capacity of the knapsack

    @model_validator(mode="after")
    def check_lengths(cls, v):
        if len(v.weights) != len(v.values):
            raise ValueError("Mismatch in number of weights and values.")
        return v


class KnapsackSolverConfig(BaseModel):
    time_limit: PositiveInt = 900  # Solver time limit in seconds
    opt_tol: NonNegativeFloat = 0.01  # Optimality tolerance (1% gap allowed)
    log_search_progress: bool = False  # Whether to log search progress


class KnapsackSolution(BaseModel):
    selected_items: List[int]  # Indices of the selected items
    objective: int  # Objective value of the solution
    upper_bound: float  # Upper bound of the solution


def solve_knapsack(
    instance: KnapsackInstance, config: KnapsackSolverConfig
) -> KnapsackSolution:
    model = cp_model.CpModel()
    n = len(instance.weights)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    model.Add(sum(instance.weights[i] * x[i] for i in range(n)) <= instance.capacity)
    model.Maximize(sum(instance.values[i] * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit
    solver.parameters.relative_gap_limit = config.opt_tol
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return KnapsackSolution(
            selected_items=[i for i in range(n) if solver.Value(x[i])],
            objective=solver.ObjectiveValue(),
            upper_bound=solver.BestObjectiveBound(),
        )
    return KnapsackSolution(selected_items=[], objective=0, upper_bound=0)
```

**Key Benefits:**

- **Structured Data Handling**: Defining explicit structures for each aspect of
  the problem ensures robust data handling and minimizes errors, facilitating
  easier integration and API exposition.
- **Easy Serialization**: Pydantic models support straightforward conversion to
  and from JSON, simplifying the storage and transmission of configurations and
  results.
- **Enhanced Testing and Documentation**: Clear data definitions make it easier
  to generate documentation and conduct tests that confirm the model and
  solver's behavior.
- **Backward Compatibility**: Default values in data classes enable seamless
  integration of older configurations with new software versions, accommodating
  new parameters without disrupting existing setups.

One challenge I often face is designing data classes to be as generic as
possible so that they can be used with multiple solvers and remain compatible
throughout various stages of the optimization process. For instance, a graph
might be represented as an edge list, an adjacency matrix, or an adjacency list,
each with its own pros and cons, complicating the decision of which format is
optimal for all stages. However, converting between different data class formats
is typically straightforward, often requiring only a few lines of code and
having a negligible impact compared to the optimization process itself.
Therefore, I recommend focusing on functionality with your current solver
without overcomplicating this aspect. There is little harm in having to call a
few convert functions because you created separate specialized data classes.

### Solver Class

In many real-world optimization scenarios, problems may require iterative
refinement of the model and solution. For instance, new constraints might only
become apparent after presenting an initial solution to a user or another
algorithm. In such cases, flexibility is crucial, making it beneficial to
encapsulate both the model and the solver within a single class. This setup
facilitates the dynamic addition of constraints and subsequent re-solving
without needing to rebuild the entire model.

**Implemented Changes:** We introduce the `KnapsackSolver` class, which
encapsulates the entire setup and solving process of the knapsack problem:

```python
class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self.n = len(instance.weights)
        self.x = [self.model.NewBoolVar(f"x_{i}") for i in range(self.n)]
        self._build_model()
        self.solver = cp_model.CpSolver()

    def _add_constraints(self):
        used_weight = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.Add(used_weight <= self.instance.capacity)

    def _add_objective(self):
        self.model.Maximize(
            sum(value * x_i for value, x_i in zip(self.instance.values, self.x))
        )

    def _build_model(self):
        self._add_constraints()
        self._add_objective()

    def solve(self) -> KnapsackSolution:
        self.solver.parameters.max_time_in_seconds = self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=[
                    i for i in range(self.n) if self.solver.Value(self.x[i])
                ],
                objective=self.solver.ObjectiveValue(),
                upper_bound=self.solver.BestObjectiveBound(),
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def prohibit_combination(self, item_a: int, item_b: int):
        self.model.Add(self.x[item_a] + self.x[item_b] <= 1)


if __name__ == "__main__":
    instance = KnapsackInstance(weights=[1, 2, 3], values=[4, 5, 6], capacity=3)
    config = KnapsackSolverConfig(time_limit=10, opt_tol=0.01, log_search_progress=True)
    solver = KnapsackSolver(instance, config)
    solution = solver.solve()
    print(solution)
    solver.prohibit_combination(0, 1)
    solution = solver.solve()
    print(solution)
```

**Key Benefits:**

- **Incremental Model Building and Re-solving**: The class structure not only
  facilitates incremental additions of constraints for iterative model
  modifications without starting from scratch but also supports multiple
  invocations of the `solve` method. This allows for iterative refinement of the
  solution by adjusting constraints or solver parameters such as time limits and
  optimality tolerance.
- **Direct Model and Solver Access**: Provides direct access to the model and
  solver, enhancing flexibility for advanced operations and debugging, a
  capability not exposed in the function-based approach.

### Variable Containers

Modularization is crucial in software engineering for maintaining and scaling
complex codebases. In the context of optimization models, modularizing by
separating variables from the core model logic is a strategic approach. This
separation facilitates easier management of variables and provides methods for
more structured interactions with them.

**Implemented Changes:** We introduce the `_ItemVariables` class, which acts as
a container for the decision variables associated with the knapsack items. This
class not only creates these variables but also offers several utility methods
to interact with them, improving the clarity and maintainability of the code.

```python
from typing import Generator, Tuple


class _ItemVariables:
    def __init__(self, instance: KnapsackInstance, model: cp_model.CpModel):
        self.instance = instance
        self.x = [model.NewBoolVar(f"x_{i}") for i in range(len(instance.weights))]

    def __getitem__(self, i):
        return self.x[i]

    def extract_packed_items(self, solver: cp_model.CpSolver) -> List[int]:
        return [i for i, x_i in enumerate(self.x) if solver.Value(x_i)]

    def used_weight(self) -> cp_model.LinearExpr:
        return sum(weight * x_i for weight, x_i in zip(self.instance.weights, self.x))

    def packed_value(self) -> cp_model.LinearExpr:
        return sum(value * x_i for value, x_i in zip(self.instance.values, self.x))

    def iter_items(
        self,
        weight_lb: float = 0.0,
        weight_ub: float = float("inf"),
        value_lb: float = 0.0,
        value_ub: float = float("inf"),
    ) -> Generator[Tuple[int, cp_model.BoolVar], None, None]:
        for i, (weight, x_i) in enumerate(zip(self.instance.weights, self.x)):
            if (
                weight_lb <= weight <= weight_ub
                and value_lb <= self.instance.values[i] <= value_ub
            ):
                yield i, x_i


class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self._item_vars = _ItemVariables(instance, self.model)
        self._build_model()
        self.solver = cp_model.CpSolver()

    def _add_constraints(self):
        self.model.Add(self._item_vars.used_weight() <= self.instance.capacity)

    def _add_objective(self):
        self.model.Maximize(self._item_vars.packed_value())

    def _build_model(self):
        self._add_constraints()
        self._add_objective()

    def solve(self) -> KnapsackSolution:
        self.solver.parameters.max_time_in_seconds = self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=self._item_vars.extract_packed_items(self.solver),
                objective=self.solver.ObjectiveValue(),
                upper_bound=self.solver.BestObjectiveBound(),
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def prohibit_combination(self, item_a: int, item_b: int):
        self.model.Add(self._item_vars[item_a] + self._item_vars[item_b] <= 1)
```

**Key Benefits:**

- **Enhanced Readability**: By encapsulating the item variables and their
  interactions within a dedicated class, the main solver class becomes more
  focused and easier to understand.
- **Improved Modularity**: The `_ItemVariables`-class allows to easily hand over
  the variables to a function that may create complex constraints, without
  creating a cyclic dependency. The model can now be split over multiple files
  without any issues.

### Submodels

As optimization models increase in complexity, it may be beneficial to divide
the overall model into smaller, more manageable submodels. These submodels can
encapsulate specific parts of the problem, communicating with the main model via
shared variables but hiding internal details like auxiliary variables.

For instance, piecewise linear functions can be modeled as submodels, as done
for `PiecewiseLinearConstraint` in
[./utils/piecewise_functions/piecewise_linear_function.py](https://github.com/d-krupke/cpsat-primer/blob/main/utils/piecewise_functions/piecewise_linear_function.py).
Each submodel handles a piecewise linear function independently, interfacing
with the main model through shared `x` and `y` variables. By encapsulating the
logic for each piecewise function in a dedicated class, we standardize and reuse
the logic across multiple instances, enhancing modularity and maintainability.

```python
requirements_1 = (3, 5, 2)
requirements_2 = (2, 1, 3)

from ortools.sat.python import cp_model

model = cp_model.CpModel()
buy_1 = model.NewIntVar(0, 1_500, "buy_1")
buy_2 = model.NewIntVar(0, 1_500, "buy_2")
buy_3 = model.NewIntVar(0, 1_500, "buy_3")

produce_1 = model.NewIntVar(0, 300, "produce_1")
produce_2 = model.NewIntVar(0, 300, "produce_2")

model.Add(produce_1 * requirements_1[0] + produce_2 * requirements_2[0] <= buy_1)
model.Add(produce_1 * requirements_1[1] + produce_2 * requirements_2[1] <= buy_2)
model.Add(produce_1 * requirements_1[2] + produce_2 * requirements_2[2] <= buy_3)

# You can find this code it ./utils!
from piecewise_functions import PiecewiseLinearFunction, PiecewiseLinearConstraint

# Define the functions for the costs
costs_1 = [(0, 0), (1000, 400), (1500, 1300)]
costs_2 = [(0, 0), (300, 300), (700, 500), (1200, 600), (1500, 1100)]
costs_3 = [(0, 0), (200, 400), (500, 700), (1000, 900), (1500, 1500)]
# PiecewiseLinearFunction is a pydantic model and can be serialized easily!
f_costs_1 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_1], ys=[y for x, y in costs_1]
)
f_costs_2 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_2], ys=[y for x, y in costs_2]
)
f_costs_3 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_3], ys=[y for x, y in costs_3]
)

# Define the functions for the gain
gain_1 = [(0, 0), (100, 800), (200, 1600), (300, 2_000)]
gain_2 = [(0, 0), (80, 1_000), (150, 1_300), (200, 1_400), (300, 1_500)]
f_gain_1 = PiecewiseLinearFunction(xs=[x for x, y in gain_1], ys=[y for x, y in gain_1])
f_gain_2 = PiecewiseLinearFunction(xs=[x for x, y in gain_2], ys=[y for x, y in gain_2])

# Create y>=f(x) constraints for the costs
x_costs_1 = PiecewiseLinearConstraint(model, buy_1, f_costs_1, upper_bound=False)
x_costs_2 = PiecewiseLinearConstraint(model, buy_2, f_costs_2, upper_bound=False)
x_costs_3 = PiecewiseLinearConstraint(model, buy_3, f_costs_3, upper_bound=False)

# Create y<=f(x) constraints for the gain
x_gain_1 = PiecewiseLinearConstraint(model, produce_1, f_gain_1, upper_bound=True)
x_gain_2 = PiecewiseLinearConstraint(model, produce_2, f_gain_2, upper_bound=True)

# Maximize the gain minus the costs
model.Maximize(x_gain_1.y + x_gain_2.y - (x_costs_1.y + x_costs_2.y + x_costs_3.y))
```

**Key Benefits:**

- **Testing**: Testing complex optimization models is often very difficult as
  the outputs are often sensitive to small changes in the model. Even if you
  have a good test case with predictable results, detected errors may be very
  difficult to track down. If you extracted elements into submodels, you can
  test these submodels independently, ensuring that they work correctly before
  integrating them into the main model.
  ```python
  def test_piecewise_linear_upper_bound_constraint():
      model = cp_model.CpModel()
      # Defining the input. Note that for some problems it may be
      # easier to fix variables to a specific value and then just
      # test feasibility.
      x = model.NewIntVar(0, 20, "x")
      f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
      # Using the submodel
      c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
      model.Maximize(c.y)
      # Checking its behavior
      solver = cp_model.CpSolver()
      assert solver.Solve(model) == cp_model.OPTIMAL
      assert solver.Value(c.y) == 10
      assert solver.Value(x) == 10
  ```
- **Modularity**: Submodels allow for the encapsulation of complex logic into
  smaller, more manageable components, enhancing code organization and
  readability.
- **Reusability**: By defining submodels for common functions or constraints,
  you can reuse these components across multiple instances of the main model,
  promoting code reuse and reducing redundancy.
- **Abstraction**: Submodels abstract the internal details of specific
  functions, enabling users to interact with them at a higher level without
  needing to understand the underlying implementation.

### Lazy Variable Construction

In models with numerous auxiliary variables, often only a subset is actually
used by the constraints. You can now try to only create the variables that may
actually be needed later on, but this can require some complex code to ensure
that exactly the right variables are created. If the model is extended later on,
things can get even more complicated as you may not know which variables are
needed upfront. This is where lazy variable construction comes into play. Here,
we create variables only when they are accessed, ensuring that only necessary
variables are generated, reducing memory usage and computational overhead. While
this can be more expensive that just creating a vector with all variables, when
in the end most variables are needed anyway, but it can save a lot of memory and
computation time if only a small subset is actually used.

**Implemented Changes:** We introduce the new class `_CombiVariables` that
manages auxiliary variables indicating that a pair of items were packed,
allowing to give additional bonuses for packing certain items together.
Theoretically, there is a square number of possible combinations, but there will
probably only be a handful of them that are actually used. By creating the
variables only when they are accessed, we can reduce memory usage and
computational overhead.

```python
class _ItemVariables:
    def __init__(self, instance: KnapsackInstance, model: cp_model.CpModel):
        self.instance = instance
        self.x = [model.NewBoolVar(f"x_{i}") for i in range(len(instance.weights))]

    def __getitem__(self, i):
        return self.x[i]

    def extract_packed_items(self, solver: cp_model.CpSolver) -> List[int]:
        return [i for i, x_i in enumerate(self.x) if solver.Value(x_i)]

    def used_weight(self) -> cp_model.LinearExpr:
        return sum(weight * x_i for weight, x_i in zip(self.instance.weights, self.x))

    def packed_value(self) -> cp_model.LinearExpr:
        return sum(value * x_i for value, x_i in zip(self.instance.values, self.x))

    def iter_items(
        self,
        weight_lb: float = 0.0,
        weight_ub: float = float("inf"),
        value_lb: float = 0.0,
        value_ub: float = float("inf"),
    ) -> Generator[Tuple[int, cp_model.BoolVar], None, None]:
        for i, (weight, x_i) in enumerate(zip(self.instance.weights, self.x)):
            if (
                weight_lb <= weight <= weight_ub
                and value_lb <= self.instance.values[i] <= value_ub
            ):
                yield i, x_i


class _CombiVariables:
    def __init__(
        self,
        instance: KnapsackInstance,
        model: cp_model.CpModel,
        item_vars: _ItemVariables,
    ):
        self.instance = instance
        self.model = model
        self.item_vars = item_vars
        self.bonus = {}

    def __getitem__(self, i, j):
        i, j = min(i, j), max(i, j)
        if (i, j) not in self.bonus:
            self.bonus[(i, j)] = self.model.NewBoolVar(f"bonus_{i}_{j}")
            self.model.Add(
                self.item_vars[i] + self.item_vars[j] >= 2 * self.bonus[(i, j)]
            )
        return self.bonus[(i, j)]


class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self._item_vars = _ItemVariables(instance, self.model)
        self._bonus_vars = _CombiVariables(instance, self.model, self._item_vars)
        self._objective = self._item_vars.packed_value()  # Initial objective setup
        self.solver = cp_model.CpSolver()

    def solve(self) -> KnapsackSolution:
        self.model.Maximize(self._objective)
        self.solver.parameters.max_time_in_seconds = self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=self._item_vars.extract_packed_items(self.solver),
                objective=self.solver.ObjectiveValue(),
                upper_bound=self.solver.BestObjectiveBound(),
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def add_bonus(self, item_a: int, item_b: int, bonus: int):
        self._objective += bonus * self._bonus_vars[item_a, item_b]
```

**Key Benefits:**

- **Efficiency**: Lazy construction of variables ensures that only necessary
  variables are created, reducing memory usage and computational overhead.
- **Simplicity**: By just creating the variables when accessed, we do not need
  any logic to decide which variables are needed upfront, simplifying the model
  construction process.

### Embedding CP-SAT in an Application via multiprocessing

If you want to embed CP-SAT in your application for potentially long-running
optimization tasks, you can utilize callbacks to provide users with progress
updates and potentially interrupt the process early. However, one issue is that
the application can only react during the callback. Since the callback is not
always called frequently, this may lead to problematic delays, making it
unsuitable for graphical user interfaces (GUIs) or application programming
interfaces (APIs).

An alternative is to let the solver run in a separate process and communicate
with it using a pipe. This approach allows the solver to be interrupted at any
time, enabling the application to react immediately. Python's multiprocessing
module provides reasonably simple ways to achieve this.
[This example](https://github.com/d-krupke/cpsat-primer/blob/main//examples/embedding_cpsat/)
showcases such an approach. However, for scaling this approach up, you will
actually have to build a task queues where the solver is run by workers. Using
multiprocessing can still be useful for the worker to remain responsive for stop
signals while the solver is running.

| ![Interactive Solver with Streamlit using multiprocessing](https://github.com/d-krupke/cpsat-primer/blob/main/images/streamlit_solver.gif) |
| :----------------------------------------------------------------------------------------------------------------------------------------: |
|                                _Using multiprocessing, one can build a responsive interface for a solver._                                 |

---
