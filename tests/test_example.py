from ortools.sat.python import cp_model


def test_model_solution():
    model = cp_model.CpModel()

    # Variables
    x = model.new_int_var(0, 100, "x")
    y = model.new_int_var(0, 100, "y")

    # Constraints
    model.add(x + y <= 30)

    # Objective
    model.maximize(30 * x + 50 * y)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    # The status tells us if we were able to compute an optimal solution.
    assert status == cp_model.OPTIMAL

    # The value of the variables in the optimal solution.
    assert solver.value(x) == 0
    assert solver.value(y) == 30
