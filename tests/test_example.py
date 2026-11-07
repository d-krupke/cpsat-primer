"""Check the introductory example and a pitfall mentioned next to it."""

import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model


def test_model_solution():
    """The first model is solved to optimality with x=0 and y=30 (value 1500)."""
    namespace = run_snippet("example/first_model")
    solver = namespace["solver"]
    assert namespace["status_code"] == cp_model.OPTIMAL
    assert solver.value(namespace["x"]) == 0
    assert solver.value(namespace["y"]) == 30
    assert solver.objective_value == 1500


def test_not_implemented_error():
    """Expressions cannot be evaluated in Python conditions."""
    model = cp_model.CpModel()
    x = model.new_int_var(0, 100, "x")

    with pytest.raises(NotImplementedError):
        if x + 1 <= 1:
            print("True")
