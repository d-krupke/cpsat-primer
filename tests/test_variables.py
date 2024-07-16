from ortools.sat.python import cp_model
import pytest
import pandas as pd


def test_basic_variables():
    model = cp_model.CpModel()
    # integer variable z with bounds -100 <= z <= 100
    z = model.new_int_var(-100, 100, "z")  # noqa: F841
    z_ = model.NewIntVar(-100, 100, "z_")  # old syntax  # noqa: F841
    # boolean variable b
    b = model.new_bool_var("b")
    b_ = model.NewBoolVar("b_")  # old syntax  # noqa: F841
    # implicitly available negation of b:
    not_b = ~b  # will be 1 if b is 0 and 0 if b is 1  # noqa: F841
    not_b_ = b.Not()  # old syntax  # noqa: F841


def test_variables_series():
    model = cp_model.CpModel()
    # list of integer variables
    index = pd.Index(range(10), name="index")
    xs = model.new_int_var_series("x", index, 0, 100)  # noqa: F841
    assert len(xs) == 10
    assert isinstance(xs, pd.Series)

    # list of boolean variables
    df = pd.DataFrame(
        data={"weight": [1 for _ in range(10)], "value": [3 for _ in range(10)]},
        index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    bs = model.new_bool_var_series("b", df.index)  # noqa: F841
    # bs is a pandas Series with boolean variables, indexed by the index of the DataFrame
    # this allows us to easily sum over the variables and multiply them with the values in the DataFrame
    assert len(bs) == 10
    assert isinstance(bs, pd.Series)
    model.add(bs @ df["weight"] <= 100)
    model.maximize(bs @ df["value"])


def test_domain_variables():
    model = cp_model.CpModel()
    # Define a domain with selected values
    domain = cp_model.Domain.from_values([2, 5, 8, 10, 20, 50, 90])
    # cam also be done via intervals
    domain_2 = cp_model.Domain.from_intervals([[8, 12], [14, 20]])

    # there are also some operations available
    domain_3 = domain.union_with(domain_2)  # noqa: F841

    # Create a domain variable within this defined domain
    x = model.new_int_var_from_domain(domain, "x")  # noqa: F841


def test_variable_without_name():
    model = cp_model.CpModel()
    # will throw a TypeError because the name is missing
    with pytest.raises(TypeError):
        x = model.new_int_var(0, 100)  # noqa: F841
