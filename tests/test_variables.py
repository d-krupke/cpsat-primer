"""Check the variable snippets of the modelling chapter."""

import pandas as pd
import pytest
from _snippets import run_snippet
from _solve import all_values
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
    values = all_values(namespace["model"], namespace["x"])
    assert values == {2, 5, 8, 10, 20, 50, 90}


def test_variable_without_name():
    """Names are mandatory; the chapter relies on them for debugging output."""
    model = cp_model.CpModel()
    with pytest.raises(TypeError):
        model.new_int_var(0, 100)  # ty: ignore[missing-argument]
