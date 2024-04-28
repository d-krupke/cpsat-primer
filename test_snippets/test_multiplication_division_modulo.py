from ortools.sat.python import cp_model

def test_multiplication_division_modulo():
    model = cp_model.CpModel()

    x = model.NewIntVar(-100, 100, "x")
    y = model.NewIntVar(-100, 100, "y")
    z = model.NewIntVar(-100, 100, "z")
    xyz = model.NewIntVar(-100*100*100, 100**3, "x*y*z")

    model.AddMultiplicationEquality(xyz, x, y, z)  # xyz = x*y*z
    model.AddModuloEquality(x, y, 3)  # x = y % 3
    model.AddDivisionEquality(x, y, z)  # x = y // z

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), \
    "cpsat cannot find a solution for the given model or the model is infeasible"
