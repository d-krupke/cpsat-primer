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

def test_cpsat_integer_vars(integer_var):  
    z, z_value = integer_var

    # check if z_variable is an integer
    assert isinstance(z_value, int), "expected an integer, but got different type"

    # Check if z has desired lower and upper bounds
    assert z.Proto().domain[0] == -100, "integer variable's lowerbound is incorrect"
    assert z.Proto().domain[1] == 100, "integer variable's upperbound is incorrect"
     