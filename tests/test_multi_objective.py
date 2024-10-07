from math import ceil, floor
from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeFloat,
    PositiveFloat,
    Field,
    model_validator,
)
from ortools.sat.python import cp_model


class KnapsackInstance(BaseModel):
    # Defines the knapsack instance to be solved.
    weights: list[PositiveInt] = Field(..., description="The weight of each item.")
    values: list[PositiveInt] = Field(..., description="The value of each item.")
    capacity: PositiveInt = Field(..., description="The capacity of the knapsack.")

    @model_validator(mode="after")
    def check_lengths(cls, v):
        if len(v.weights) != len(v.values):
            raise ValueError("Mismatch in number of weights and values.")
        return v


class KnapsackSolverConfig(BaseModel):
    # Defines the configuration for the knapsack solver.
    time_limit: PositiveFloat = Field(
        default=900.0, description="Time limit in seconds."
    )
    opt_tol: NonNegativeFloat = Field(
        default=0.01, description="Optimality tolerance (1% gap allowed)."
    )
    log_search_progress: bool = Field(
        default=False, description="Whether to log the search progress."
    )


class KnapsackSolution(BaseModel):
    # Defines the solution of the knapsack problem.
    selected_items: list[int] = Field(..., description="Indices of selected items.")
    objective: float = Field(..., description="Objective value of the solution.")
    upper_bound: float = Field(
        ...,
        description="Upper bound of the solution, i.e., a proven limit on how good a solution could be.",
    )


class MultiObjectiveKnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self.n = len(instance.weights)
        self.x = [self.model.new_bool_var(f"x_{i}") for i in range(self.n)]
        self._objective = 0
        self._build_model()
        self.solver = cp_model.CpSolver()

    def set_maximize_value_objective(self):
        """Set the objective to maximize the value of the packed goods."""
        self._objective = sum(
            value * x_i for value, x_i in zip(self.instance.values, self.x)
        )
        self.model.maximize(self._objective)

    def set_minimize_weight_objective(self):
        """Set the objective to minimize the weight of the packed goods."""
        self._objective = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.minimize(self._objective)

    def _set_solution_as_hint(self):
        """Use the current solution as a hint for the next solve."""
        for i, v in enumerate(self.model.proto.variables):
            v_ = self.model.get_int_var_from_proto_index(i)
            assert v.name == v_.name, "Variable names should match"
            self.model.add_hint(v_, self.solver.value(v_))

    def fix_current_objective(self, ratio: float = 1.0):
        """Fix the current objective value to prevent degeneration."""
        if ratio == 1.0:
            self.model.add(self._objective == self.solver.objective_value)
        elif ratio > 1.0:
            self.model.add(self._objective <= ceil(self.solver.objective_value * ratio))
        else:
            self.model.add(
                self._objective >= floor(self.solver.objective_value * ratio)
            )

    def _add_constraints(self):
        """Add the weight constraint to the model."""
        used_weight = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.add(used_weight <= self.instance.capacity)

    def _build_model(self):
        """Build the initial model with constraints and objective."""
        self._add_constraints()
        self.set_maximize_value_objective()

    def solve(self, time_limit: float | None = None) -> KnapsackSolution:
        """Solve the knapsack problem and return the solution."""
        self.solver.parameters.max_time_in_seconds = (
            time_limit if time_limit else self.config.time_limit
        )
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._set_solution_as_hint()
            return KnapsackSolution(
                selected_items=[
                    i for i in range(self.n) if self.solver.value(self.x[i])
                ],
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )


def test_multi_objective():
    instance = KnapsackInstance(
        weights=[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
        values=[92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
        capacity=165,
    )
    # Define a solver configuration
    config = KnapsackSolverConfig(
        time_limit=10.0, opt_tol=0.01, log_search_progress=False
    )
    solver = MultiObjectiveKnapsackSolver(instance, config)
    solution_1 = solver.solve()  # noqa: F841

    # maintain at least 95% of the current objective value
    solver.fix_current_objective(0.95)
    # change the objective to minimize the weight
    solver.set_minimize_weight_objective()
    solution_2 = solver.solve(time_limit=10)  # noqa: F841
