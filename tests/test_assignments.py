from ortools.sat.python import cp_model


def test_lin_expr_in_domain():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")

    # Define the domain
    domain = cp_model.Domain.from_values([20, 50, 100])

    model.add_linear_expression_in_domain(linear_expr=10 * x + 5 * y, domain=domain)


def test_allowed_assignments():
    model = cp_model.CpModel()
    x_employee_1 = model.new_bool_var("x_employee_1")
    x_employee_2 = model.new_bool_var("x_employee_2")
    x_employee_3 = model.new_bool_var("x_employee_3")
    x_employee_4 = model.new_bool_var("x_employee_4")

    # Define the allowed assignments
    allowed_assignments = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
    ]

    model.add_allowed_assignments(
        [x_employee_1, x_employee_2, x_employee_3, x_employee_4], allowed_assignments
    )


def test_forbidden_assignments():
    model = cp_model.CpModel()
    x_employee_1 = model.new_bool_var("x_employee_1")
    x_employee_2 = model.new_bool_var("x_employee_2")
    x_employee_3 = model.new_bool_var("x_employee_3")
    x_employee_4 = model.new_bool_var("x_employee_4")
    prohibit_assignments = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
    ]
    model.add_forbidden_assignments(
        [x_employee_1, x_employee_2, x_employee_3, x_employee_4], prohibit_assignments
    )
