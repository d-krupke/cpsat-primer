"""Check the element and inverse snippets of the modelling chapter."""

import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model


def status_with(
    namespace: dict, variables: list, values: list
) -> cp_model.CpSolverStatus:
    model = namespace["model"]
    for variable, value in zip(variables, values, strict=True):
        model.add(variable == value)
    return cp_model.CpSolver().solve(model)


@pytest.mark.parametrize(
    "x, y, z, index, value, feasible",
    [
        (3, 4, 5, 0, 3, True),
        (3, 4, 5, 1, 4, True),
        (3, 4, 5, 2, 5, True),
        (7, 3, 4, 0, 7, True),
        (3, 4, 5, 1, 5, False),
    ],
)
def test_element(x, y, z, index, value, feasible):
    """`value_at_index_var` equals `var_array[index_var]`."""
    namespace = run_snippet("modelling/element")
    names = ("x", "y", "z", "index_var", "value_at_index_var")
    variables = [namespace[name] for name in names]
    status = status_with(namespace, variables, [x, y, z, index, value])
    assert status == (cp_model.OPTIMAL if feasible else cp_model.INFEASIBLE)


@pytest.mark.parametrize(
    "v_values, w_values, feasible",
    [
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], True),
        ([1, 2, 3, 4, 5, 0], [5, 0, 1, 2, 3, 4], True),
        ([1, 0, 3, 5, 2, 4], [1, 0, 4, 2, 5, 3], True),
        ([1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0], False),
    ],
)
def test_inverse(v_values, w_values, feasible):
    """`v[i] == j` holds exactly if `w[j] == i`."""
    namespace = run_snippet("modelling/inverse")
    variables = [*namespace["v"], *namespace["w"]]
    status = status_with(namespace, variables, v_values + w_values)
    assert status == (cp_model.OPTIMAL if feasible else cp_model.INFEASIBLE)
