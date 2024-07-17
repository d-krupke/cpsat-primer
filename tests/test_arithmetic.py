from ortools.sat.python import cp_model


def test_arithmetic():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")

    xyz = model.new_int_var(-(100**3), 100**3, "x*y*z")
    model.add_multiplication_equality(xyz, [x, y, z])  # xyz = x*y*z

    model.add_modulo_equality(x, y, 3)  # x = y % 3
    model.add_division_equality(x, y, z)  # x = y // z

    solver = cp_model.CpSolver()
    solver.solve(model)
