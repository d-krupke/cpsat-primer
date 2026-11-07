"""Solve helpers shared by the tests that check what chapter snippets compute.

Created in November 2026 so solver settings live in one place: the models are
tiny, and a single worker keeps runs deterministic and independent of the
machine's core count.
"""

from collections.abc import Iterable
from typing import Any

from ortools.sat.python import cp_model


def make_solver() -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 3
    return solver


def solve(model: cp_model.CpModel) -> tuple[cp_model.CpSolverStatus, cp_model.CpSolver]:
    solver = make_solver()
    return solver.solve(model), solver


def status_with(
    model: cp_model.CpModel, variables: Iterable[Any], values: Iterable[int]
) -> cp_model.CpSolverStatus:
    """Fix the variables to the values and report whether that is feasible."""
    for variable, value in zip(variables, values, strict=True):
        model.add(variable == value)
    return solve(model)[0]


def all_values(model: cp_model.CpModel, expression: Any) -> set[int]:
    """Every value the expression takes over all solutions of the model."""
    solver = make_solver()
    solver.parameters.enumerate_all_solutions = True
    values: set[int] = set()

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            values.add(self.value(expression))

    solver.solve(model, Collect())
    return values
