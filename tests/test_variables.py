from ortools.sat.python import cp_model
import pytest


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


def test_domain_variables():
    model = cp_model.CpModel()
    # Define a domain with selected values
    domain = cp_model.Domain.from_values([2, 5, 8, 10, 20, 50, 90])
    # cam also be done via intervals
    domain_2 = cp_model.Domain.from_intervals([[8, 12], [14, 20]])

    # there are also some operations available
    domain_3 = domain.union_with(domain_2)  # noqa: F841

    # Create a domain variable within this defined domain
    x = model.NewIntVarFromDomain(domain, "x")  # noqa: F841


def test_variable_without_name():
    model = cp_model.CpModel()
    # will throw a TypeError because the name is missing
    with pytest.raises(TypeError):
        x = model.new_int_var(0, 100)  # noqa: F841
