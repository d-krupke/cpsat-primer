from ortools.sat.python import cp_model


def test_element():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")
    var_array = [x, y, z]

    # Create a variable for the index and a variable for the value at that index.
    index_var = model.new_int_var(0, len(var_array) - 1, "index")
    value_at_index_var = model.new_int_var(-100, 100, "value_at_index")

    # Bind the variables together with the element constraint.
    model.add_element(expressions=var_array, index=index_var, target=value_at_index_var)


def test_feasible_assignments():
    # List of test cases for feasible assignments
    test_cases = [(3, 4, 5, 0, 3), (3, 4, 5, 1, 4), (3, 4, 5, 2, 5), (7, 3, 4, 0, 7)]

    # Test each case
    for test_case in test_cases:
        # Reset the model constraints for the variables
        solver = cp_model.CpSolver()
        # Initialize the CP model
        model = cp_model.CpModel()
        x = model.new_int_var(-100, 100, "x")
        y = model.new_int_var(-100, 100, "y")
        z = model.new_int_var(-100, 100, "z")
        var_array = [x, y, z]

        # Create a variable for the index and a variable for the value at that index.
        index_var = model.new_int_var(0, len(var_array) - 1, "index")
        value_at_index_var = model.new_int_var(-100, 100, "value_at_index")

        # Bind the variables together with the element constraint.
        model.add_element(
            expressions=var_array, index=index_var, target=value_at_index_var
        )

        # Apply the test case values
        x_value, y_value, z_value, index_value, value_at_index_value = test_case
        model.add(x == x_value)
        model.add(y == y_value)
        model.add(z == z_value)
        model.add(index_var == index_value)
        model.add(value_at_index_var == value_at_index_value)

        # Solve the model
        status = solver.solve(model)

        # Check the solution
        assert status == cp_model.OPTIMAL


def test_inverse_constraint():
    # Test cases for feasible assignments
    test_cases = [
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5, 0], [5, 0, 1, 2, 3, 4]),
        ([1, 0, 3, 5, 2, 4], [1, 0, 4, 2, 5, 3]),
    ]

    # Test each case
    for v_values, w_values in test_cases:
        print(v_values, w_values)
        solver = cp_model.CpSolver()
        model = cp_model.CpModel()

        # Define variables
        v = [model.new_int_var(0, 5, f"v_{i}") for i in range(6)]
        w = [model.new_int_var(0, 5, f"w_{i}") for i in range(6)]

        # Add the inverse constraint
        model.add_inverse(v, w)

        # Set the values of v and w according to the test case
        for i in range(6):
            model.add(v[i] == v_values[i])
            model.add(w[i] == w_values[i])

        # Solve the model
        status = solver.solve(model)

        assert (
            status == cp_model.OPTIMAL
        ), "The model did not solve to optimal even though it should be feasible."
