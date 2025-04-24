from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeFloat,
    PositiveFloat,
    Field,
    model_validator,
)
from datetime import datetime
from hashlib import md5
from pathlib import Path
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


def solve_knapsack(
    instance: KnapsackInstance, config: KnapsackSolverConfig
) -> KnapsackSolution:
    model = cp_model.CpModel()
    n = len(instance.weights)
    x = [model.new_bool_var(f"x_{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) <= instance.capacity)
    model.maximize(sum(instance.values[i] * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    # Set solver parameters from the configuration
    solver.parameters.max_time_in_seconds = config.time_limit
    solver.parameters.relative_gap_limit = config.opt_tol
    solver.parameters.log_search_progress = config.log_search_progress
    # solve the model and return the solution
    status = solver.solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return KnapsackSolution(
            selected_items=[i for i in range(n) if solver.value(x[i])],
            objective=solver.objective_value,
            upper_bound=solver.best_objective_bound,
        )
    return KnapsackSolution(selected_items=[], objective=0, upper_bound=0)


def add_test_case(instance: KnapsackInstance, config: KnapsackSolverConfig):
    """
    Quickly generate a test case based on the instance and configuration.
    Be aware that the difficult models that are
    """
    test_folder = Path(__file__).parent / "test_data"
    unique_id = (
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_"
        + md5(
            (instance.model_dump_json() + config.model_dump_json()).encode()
        ).hexdigest()
    )
    subfolder = test_folder / "knapsack" / unique_id
    subfolder.mkdir(parents=True, exist_ok=True)
    with open(subfolder / "instance.json", "w") as f:
        f.write(instance.model_dump_json())
    with open(subfolder / "config.json", "w") as f:
        f.write(config.model_dump_json())
    solution = solve_knapsack(instance, config)
    with open(subfolder / "solution.json", "w") as f:
        f.write(solution.model_dump_json())


def test_saved_test_cases():
    test_folder = Path(__file__).parent / "test_data"
    for subfolder in test_folder.glob("knapsack/*"):
        with open(subfolder / "instance.json") as f:
            instance = KnapsackInstance.model_validate_json(f.read())
        with open(subfolder / "config.json") as f:
            config = KnapsackSolverConfig.model_validate_json(f.read())
        with open(subfolder / "solution.json") as f:
            solution = KnapsackSolution.model_validate_json(f.read())
        new_solution = solve_knapsack(instance, config)
        assert new_solution.objective <= solution.upper_bound, (
            "New solution is better than the previous upper bound: One has to be wrong."
        )
        assert solution.objective <= new_solution.upper_bound, (
            "Old solution is better than the new upper bound: One has to be wrong."
        )
        # Do not test for the selected items, as the solver might return a different solution of the same quality


if __name__ == "__main__":
    # Define a knapsack instance
    instance = KnapsackInstance(
        weights=[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
        values=[92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
        capacity=165,
    )
    # Define a solver configuration
    config = KnapsackSolverConfig(
        time_limit=10.0, opt_tol=0.01, log_search_progress=False
    )
    # Solve the knapsack problem
    solution = solve_knapsack(instance, config)
    # Add the test case to the test data folder
    add_test_case(instance, config)
