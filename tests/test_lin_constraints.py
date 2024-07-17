from ortools.sat.python import cp_model


def test_lin_constraints():
    model = cp_model.CpModel()
    # integer variable z with bounds -100 <= z <= 100
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")

    model.add(10 * x + 15 * y <= 10)
    model.add(x + z == 2 * y)

    # This one actually is not linear but still works.
    model.add(x + y != z)

    # For <, > you can simply use <= and -1 because we are working on integers.
    model.add(x <= z - 1)  # x < z

    model.add(x < y + z)
    model.add(y > 300 - 4 * z)


def test_infeasible_intersection():
    model = cp_model.CpModel()
    # integer variable z with bounds -100 <= z <= 100
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")

    model.add(x - y == 0)
    model.add(4 - x == 2 * y)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.INFEASIBLE


def test_add_linear_constraint():
    model = cp_model.CpModel()
    # integer variable z with bounds -100 <= z <= 100
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    model.add_linear_constraint(linear_expr=10 * x + 15 * y, lb=-100, ub=10)
