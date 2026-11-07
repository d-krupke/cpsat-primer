"""Check what the all-different snippets of the modelling chapter promise."""

from _snippets import run_snippet
from ortools.sat.python import cp_model


def test_all_different_with_transformations():
    """Both the plain and the shifted (`v_i + i`) constraint hold in a solution."""
    namespace = run_snippet("modelling/all_different")
    solver = cp_model.CpSolver()
    assert solver.solve(namespace["model"]) == cp_model.OPTIMAL
    xyz = [solver.value(namespace[name]) for name in ("x", "y", "z")]
    assert len(set(xyz)) == 3
    shifted = [solver.value(v) + i for i, v in enumerate(namespace["vars"])]
    assert len(set(shifted)) == len(shifted)
