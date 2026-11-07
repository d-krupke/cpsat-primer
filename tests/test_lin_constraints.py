"""Check the linear-constraint snippets of the modelling chapter."""

from _snippets import run_snippet
from _solve import solve
from ortools.sat.python import cp_model


def test_linear_constraints_hold():
    """A solution satisfies all constraints, including the strict `<` and `>`."""
    namespace = run_snippet("modelling/linear_constraints")
    status, solver = solve(namespace["model"])
    assert status == cp_model.OPTIMAL
    x, y, z = (solver.value(namespace[name]) for name in ("x", "y", "z"))
    assert 10 * x + 15 * y <= 10
    assert x + z == 2 * y
    assert x + y != z
    assert x < y + z
    assert y > 300 - 4 * z


def test_lin_constraint_sum():
    """`LinearExpr.sum` and `weighted_sum` build the same expressions as `sum`."""
    model = cp_model.CpModel()
    vars = [model.new_int_var(0, 10, f"x{i}") for i in range(5)]
    weights = [i for i in range(5)]
    model.add(cp_model.LinearExpr.sum(vars) >= 1)
    model.minimize(cp_model.LinearExpr.weighted_sum(vars, weights))
    status, solver = solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.objective_value == 0  # x0 has weight 0
