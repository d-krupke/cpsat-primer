from ortools.sat.python import cp_model


def test_objective():
    model = cp_model.CpModel()
    # integer variable z with bounds -100 <= z <= 100
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    model.add(x + 10 * y <= 100)

    model.maximize(30 * x + 50 * y)


def test_sum_objective():
    model = cp_model.CpModel()
    x_vars = [model.new_bool_var(f"x{i}") for i in range(10)]
    model.minimize(
        sum(i * x_vars[i] if i % 2 == 0 else i * ~x_vars[i] for i in range(10))
    )


def test_lexicographic_optimization():
    # some basic model
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")
    model.add(x + 10 * y - 2 * z <= 100)

    # Define the objectives
    first_objective = 30 * x + 50 * y
    second_objective = 10 * x + 20 * y + 30 * z

    # Optimize for the first objective
    model.maximize(first_objective)
    solver = cp_model.CpSolver()
    solver.solve(model)

    # Fix the first objective and optimize for the second
    model.add(first_objective == int(solver.objective_value))  # fix previous objective
    model.minimize(second_objective)  # optimize for second objective
    solver.solve(model)


def test_aux_var_objective():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")

    abs_x = model.new_int_var(0, 100, "|x|")
    model.add_abs_equality(target=abs_x, expr=x)
    model.minimize(abs_x)
