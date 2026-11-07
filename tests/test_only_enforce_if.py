"""Check that the `only_enforce_if` snippets enforce what the text says."""

import pytest
from _snippets import run_snippet
from _solve import solve
from ortools.sat.python import cp_model


@pytest.mark.parametrize(
    ("load", "has_license", "expected_cost"),
    [
        (1, False, 30),
        (50, False, 30),
        (51, False, 40),
        (80, False, 40),
        (81, True, 80),
        (100, True, 80),
        (81, False, None),
    ],
)
def test_truck_example(load, has_license, expected_cost):
    """Exactly one truck is rented, the cheapest one whose limit fits the load;
    truck c (above 80) needs a license, so without one the load is infeasible."""
    namespace = run_snippet("modelling/truck")
    model = namespace["model"]
    model.add(namespace["load_value"] == load)
    model.add(namespace["driver_has_big_truck_license"] == has_license)
    model.add(namespace["driver_has_special_license"] == 0)
    status, solver = solve(model)
    if expected_cost is None:
        assert status == cp_model.INFEASIBLE
    else:
        assert status == cp_model.OPTIMAL
        assert solver.objective_value == expected_cost
        trucks = [namespace[f"truck_{t}"] for t in "abc"]
        assert sum(solver.value(truck) for truck in trucks) == 1


def test_conjunction_with_negation():
    """With b1 and not b2, `x + y == 10` is enforced; otherwise it is not."""
    for b2, expected in ((0, cp_model.INFEASIBLE), (1, cp_model.OPTIMAL)):
        namespace = run_snippet("modelling/enforce_conjunction")
        model = namespace["model"]
        model.add(namespace["b1"] == 1)
        model.add(namespace["b2"] == b2)
        model.add(namespace["x"] + namespace["y"] == 9)
        assert solve(model)[0] == expected
