import pytest
from ortools.sat.python import cp_model

model = cp_model.CpModel()
solver = cp_model.CpSolver()

@pytest.fixture
def integer_var():
    # integer variable z with bounds -100 <= z <= 100
    z = model.NewIntVar(-100, 100, "z")
    solver.Solve(model)
    return z, solver.Value(z)

@pytest.fixture
def bool_var():
    # boolean variable b
    b = model.NewBoolVar("b")
    solver.Solve(model)
    return b, solver.Value(b)

def test_cpsat_integer_vars(integer_var):  
    z, z_value = integer_var

    # check if z_variable is an integer
    assert isinstance(z_value, int), "expected an integer, but got different type"

    # Check if z has desired lower and upper bounds
    assert z.Proto().domain[0] == -100, "integer variable's lowerbound is incorrect"
    assert z.Proto().domain[1] == 100, "integer variable's upperbound is incorrect"

def test_cpsat_bool_vars(bool_var): 
    _, b_value = bool_var

    # check if b is either 0 or 1
    assert b_value in [0, 1], "0 or 1 is expected"

def test_cpsat_not_b(bool_var): 
    b, b_value = bool_var 

    # implicitly available negation of b:
    not_b = b.Not() # will be 1 if b is 0 and 0 if b is 1
    not_b_value = solver.Value(not_b)

    # check if not_b is either 0 or 1
    assert not_b_value in [0, 1], "expected 0 or 1, but got different value"

    # check if not_b is the negation of b
    assert not_b_value == (not b_value), "not_b must be the negation of b"

     