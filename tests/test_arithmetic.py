"""Check the arithmetic snippet against the semantics the text describes."""

import math

from _snippets import run_snippet
from ortools.sat.python import cp_model


def test_arithmetic_relations_hold():
    """The solution satisfies the product, a C-style modulo (sign of `y`) and a
    division that truncates toward zero."""
    namespace = run_snippet("modelling/arithmetic")
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 3
    assert solver.solve(namespace["model"]) == cp_model.OPTIMAL
    x, y, z, xyz = (solver.value(namespace[n]) for n in ("x", "y", "z", "xyz"))
    assert xyz == x * y * z
    assert x == math.fmod(y, 3)
    assert x == int(y / z)
