from ortools.sat.python import cp_model


def test_automaton_model():
    transition_triples = [
        (
            0,
            0,
            1,
        ),  # if we are in state 0 and the transition value is 0, we go to state 1
        (
            1,
            0,
            1,
        ),  # if we are in state 1 and the transition value is 0, stay in state 1
        (1, 1, 2),  # if we are in state 1 and the transition value is 1, go to state 2
        (2, 0, 0),  # if we are in state 2 and the transition value is 0, go to state 0
        (2, 1, 1),  # if we are in state 2 and the transition value is 1, go to state 1
        (2, 2, 3),  # if we are in state 2 and the transition value is 2, go to state 3
        (
            3,
            0,
            3,
        ),  # if we are in state 3 and the transition value is 0, stay in state 3
    ]

    test_cases = [
        {"solution": [0, 1, 2, 0], "expected_status": cp_model.OPTIMAL},
        {"solution": [1, 0, 1, 2], "expected_status": cp_model.INFEASIBLE},
        {"solution": [0, 0, 1, 1], "expected_status": cp_model.INFEASIBLE},
    ]

    for test in test_cases:
        model = cp_model.CpModel()
        transition_vars = [model.new_int_var(0, 2, f"transition_{i}") for i in range(4)]
        model.add_automaton(transition_vars, 0, [3], transition_triples)

        solution = test["solution"]
        expected_status = test["expected_status"]

        # Enforce the solution values
        for i, value in enumerate(solution):
            model.add(transition_vars[i] == value)

        solver = cp_model.CpSolver()
        status = solver.solve(model)

        assert status == expected_status, (
            f"Test failed for solution {solution}. Expected status {expected_status}, got {status}."
        )
