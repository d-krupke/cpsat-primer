"""Check the domain and table-constraint snippets of the modelling chapter."""

import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model

EMPLOYEES = [f"x_employee_{i}" for i in range(1, 5)]


def status_with(namespace: dict, names: list[str], values) -> cp_model.CpSolverStatus:
    model = namespace["model"]
    for name, value in zip(names, values, strict=True):
        model.add(namespace[name] == value)
    return cp_model.CpSolver().solve(model)


def test_linear_expression_in_domain():
    """Every solution puts `10x + 5y` into {20, 50, 100}."""
    namespace = run_snippet("modelling/linear_domain")
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    expression = 10 * namespace["x"] + 5 * namespace["y"]
    values = set()

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            values.add(self.value(expression))

    solver.solve(namespace["model"], Collect())
    assert values == {20, 50, 100}


@pytest.mark.parametrize(
    "assignment, allowed",
    [((1, 0, 1, 0), True), ((0, 1, 0, 1), True), ((1, 1, 0, 0), False)],
)
def test_allowed_assignments(assignment, allowed):
    """Only the listed tuples are feasible."""
    status = status_with(
        run_snippet("modelling/allowed_assignments"), EMPLOYEES, assignment
    )
    assert status == (cp_model.OPTIMAL if allowed else cp_model.INFEASIBLE)


@pytest.mark.parametrize(
    "assignment, allowed",
    [((1, 0, 1, 0), False), ((0, 1, 0, 1), False), ((1, 1, 0, 0), True)],
)
def test_forbidden_assignments(assignment, allowed):
    """The listed tuples are infeasible, all others remain possible."""
    status = status_with(
        run_snippet("modelling/forbidden_assignments"), EMPLOYEES, assignment
    )
    assert status == (cp_model.OPTIMAL if allowed else cp_model.INFEASIBLE)
