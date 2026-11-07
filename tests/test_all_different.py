"""Check what the all-different snippets of the modelling chapter promise."""

from _snippets import run_snippet
from _solve import solve
from ortools.sat.python import cp_model


def test_all_different_with_transformations():
    """Both the plain and the shifted (`v_i + i`) constraint hold in a solution."""
    namespace = run_snippet("modelling/all_different")
    status, solver = solve(namespace["model"])
    assert status == cp_model.OPTIMAL
    xyz = [solver.value(namespace[name]) for name in ("x", "y", "z")]
    assert len(set(xyz)) == 3
    shifted = [solver.value(v) + i for i, v in enumerate(namespace["vars"])]
    assert len(set(shifted)) == len(shifted)
