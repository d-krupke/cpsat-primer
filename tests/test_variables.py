"""Check the variable snippets of the modelling chapter."""

import pandas as pd
import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model


def test_variable_series():
    """The series helpers return pandas Series aligned with the given index."""
    namespace = run_snippet("modelling/variable_series")
    for series in (namespace["xs"], namespace["bs"]):
        assert isinstance(series, pd.Series)
        assert len(series) == 10
    assert list(namespace["bs"].index) == list(namespace["df"].index)


def test_domain_variable_takes_only_domain_values():
    """`x` can take exactly the seven listed values."""
    namespace = run_snippet("modelling/domains")
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    values = set()

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            values.add(self.value(namespace["x"]))

    solver.solve(namespace["model"], Collect())
    assert values == {2, 5, 8, 10, 20, 50, 90}


def test_variable_without_name():
    """Names are mandatory; the chapter relies on them for debugging output."""
    model = cp_model.CpModel()
    with pytest.raises(TypeError):
        model.new_int_var(0, 100)  # ty: ignore[missing-argument]
