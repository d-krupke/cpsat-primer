from ortools.sat.python import cp_model


def test_arithmetic():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var_from_domain(
        cp_model.Domain.from_intervals([[-100, -1], [1, 100]]), "z"
    )

    xyz = model.new_int_var(-(100**3), 100**3, "x*y*z")
    model.add_multiplication_equality(xyz, [x, y, z])  # xyz = x*y*z

    model.add_modulo_equality(x, y, 3)  # remainder has the sign of y
    model.add_division_equality(x, y, z)  # division truncates toward zero

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 3
    assert solver.solve(model) == cp_model.OPTIMAL
