"""Check the domain and table-constraint snippets of the modelling chapter."""

import pytest
from _snippets import run_snippet
from _solve import all_values, status_with
from ortools.sat.python import cp_model

EMPLOYEES = [f"x_employee_{i}" for i in range(1, 5)]


def employees_fixed_to(snippet_id: str, assignment) -> cp_model.CpSolverStatus:
    namespace = run_snippet(snippet_id)
    employees = [namespace[name] for name in EMPLOYEES]
    return status_with(namespace["model"], employees, assignment)


def test_linear_expression_in_domain():
    """Every solution puts `10x + 5y` into {20, 50, 100}."""
    namespace = run_snippet("modelling/linear_domain")
    expression = 10 * namespace["x"] + 5 * namespace["y"]
    assert all_values(namespace["model"], expression) == {20, 50, 100}


@pytest.mark.parametrize(
    "assignment, allowed",
    [((1, 0, 1, 0), True), ((0, 1, 0, 1), True), ((1, 1, 0, 0), False)],
)
def test_allowed_assignments(assignment, allowed):
    """Only the listed tuples are feasible."""
    status = employees_fixed_to("modelling/allowed_assignments", assignment)
    assert status == (cp_model.OPTIMAL if allowed else cp_model.INFEASIBLE)


@pytest.mark.parametrize(
    "assignment, allowed",
    [((1, 0, 1, 0), False), ((0, 1, 0, 1), False), ((1, 1, 0, 0), True)],
)
def test_forbidden_assignments(assignment, allowed):
    """The listed tuples are infeasible, all others remain possible."""
    status = employees_fixed_to("modelling/forbidden_assignments", assignment)
    assert status == (cp_model.OPTIMAL if allowed else cp_model.INFEASIBLE)
