"""Check the objective snippets of the modelling chapter."""

from _snippets import run_snippet
from _solve import solve
from ortools.sat.python import cp_model


def test_lexicographic_optimization():
    """The second solve keeps the first optimum: x=100, y=20 (z=100 makes room
    for x + 10y <= 300), i.e., 30x + 50y = 4000."""
    namespace = run_snippet("modelling/lexicographic")
    assert namespace["solver"].value(namespace["first_objective"]) == 4000


def test_abs_objective():
    """Minimizing |x| over x in [-100, 100] reaches 0."""
    namespace = run_snippet("modelling/abs_objective")
    status, solver = solve(namespace["model"])
    assert status == cp_model.OPTIMAL
    assert solver.objective_value == 0
