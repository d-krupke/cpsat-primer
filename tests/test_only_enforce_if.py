"""Check that the `only_enforce_if` snippets enforce what the text says."""

from _snippets import run_snippet
from ortools.sat.python import cp_model


def solve(model: cp_model.CpModel) -> tuple[cp_model.CpSolverStatus, cp_model.CpSolver]:
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    return solver.solve(model), solver


def test_truck_limits_the_load():
    """Renting truck a caps the load at 50, so a load of 60 is infeasible."""
    namespace = run_snippet("modelling/truck")
    model = namespace["model"]
    model.add(namespace["truck_a"] == 1)
    model.add(namespace["load_value"] == 60)
    assert solve(model)[0] == cp_model.INFEASIBLE


def test_truck_c_needs_a_license():
    """Renting truck c forces one of the two licenses."""
    namespace = run_snippet("modelling/truck")
    model = namespace["model"]
    model.add(namespace["truck_c"] == 1)
    status, solver = solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(namespace["driver_has_big_truck_license"]) or solver.value(
        namespace["driver_has_special_license"]
    )


def test_conjunction_with_negation():
    """With b1 and not b2, `x + y == 10` is enforced; otherwise it is not."""
    for b2, expected in ((0, cp_model.INFEASIBLE), (1, cp_model.OPTIMAL)):
        namespace = run_snippet("modelling/enforce_conjunction")
        model = namespace["model"]
        model.add(namespace["b1"] == 1)
        model.add(namespace["b2"] == b2)
        model.add(namespace["x"] + namespace["y"] == 9)
        assert solve(model)[0] == expected
