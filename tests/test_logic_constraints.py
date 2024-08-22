from ortools.sat.python import cp_model

import pytest


def test_bool_constraints():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    model.add_bool_or(b1, b2, b3)  # b1 or b2 or b3 (at least one)
    model.add_at_least_one([b1, b2, b3])  # alternative notation
    model.add_bool_and(b1, ~b2, ~b3)  # b1 and not b2 and not b3 (all)
    model.add_bool_and(b1, ~b2, ~b3)  # Alternative notation for `Not()`
    model.add_bool_xor(b1, b2, b3)  # b1 xor b2 xor b3
    model.add_exactly_one([b1, b2, b3])  # exactly one of them
    model.add_implication(a=b1, b=b2)  # b1 -> b2
    model.add_at_most_one(
        [b1, b2, b3]
    )  # at most one of them. This is actually a more complex constraint.


def test_one_true_xor_feasible():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    # Setting the variables according to the test case
    model.add(b1 == 1)  # True
    model.add(b2 == 0)  # False
    model.add(b3 == 0)  # False

    # XOR Constraint
    model.add_bool_xor([b1, b2, b3])

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL, "Test case for one true variable failed."


def test_two_true_xor_infeasible():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    # Setting the variables according to the test case
    model.add(b1 == 1)  # True
    model.add(b2 == 1)  # True
    model.add(b3 == 0)  # False

    # XOR Constraint
    model.add_bool_xor([b1, b2, b3])

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.INFEASIBLE, "Test case for two true variables failed."


def test_all_true_xor_feasible():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    # Setting the variables according to the test case
    model.add(b1 == 1)  # True
    model.add(b2 == 1)  # True
    model.add(b3 == 1)  # True

    # XOR Constraint
    model.add_bool_xor([b1, b2, b3])

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL, "Test case for all true variables failed."


def test_all_false_xor_infeasible():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    # Setting the variables according to the test case
    model.add(b1 == 0)  # False
    model.add(b2 == 0)  # False
    model.add(b3 == 0)  # False

    # XOR Constraint
    model.add_bool_xor([b1, b2, b3])

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.INFEASIBLE, "Test case for all false variables failed."


def test_integer_cannot_be_used_in_boolean_logic():
    """
    Integer variables cannot be used in boolean logic constraints, as of CP-SAT 9.9.
    Check that this is still the case for any future versions.
    """
    model = cp_model.CpModel()

    x = model.new_int_var(0, 100, "x")
    with pytest.raises(TypeError):
        ~x  # This should raise an error because x is an integer variable

    b1 = model.new_bool_var("b1")
    with pytest.raises(TypeError):
        model.add_bool_or(x, b1)  # This should also raise an error
