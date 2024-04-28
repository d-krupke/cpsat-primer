from ortools.sat.python import cp_model

def test_random_intvar_example(): 
    import random

    random.seed(42)
    model = cp_model.CpModel()
    # Generate a random number of variables
    num_vars = random.randint(2, 20)
    # Generate 4 random numbers and 5 lists of random coefficients 
    # to randomly create constraints and a objective function
    random_numbers = [random.randint(-1000, 1000) for _ in range(4)]
    coefs = [[random.randint(-50, 50) for _ in range(num_vars)]
            for _ in range(5)]
    
    vars = [model.NewIntVar(0, 100, f"var_{i}") for i in range(num_vars)]
    
    model.Add(sum(coefs[0][j] * vars[j] for j in range(num_vars)) <= random_numbers[0])
    model.Add(sum(coefs[1][j] * vars[j] for j in range(num_vars)) >= random_numbers[1])
    model.Add(sum(coefs[2][j] * vars[j] for j in range(num_vars)) <= random_numbers[2])
    model.Add(sum(coefs[3][j] * vars[j] for j in range(num_vars)) >= random_numbers[3])
    model.Maximize(sum(coefs[4][j] * vars[j] for j in range(num_vars)))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in {cp_model.OPTIMAL, cp_model.INFEASIBLE}, \
    "Status should be optimal or the model should be infeasible"

    if status == cp_model.OPTIMAL:
        assert sum(coefs[0][j]*solver.Value(vars[j]) for j in range(num_vars)) <= random_numbers[0], \
        "Constraint 1 must be satisfied"
        assert sum(coefs[1][j]*solver.Value(vars[j]) for j in range(num_vars)) >= random_numbers[1], \
        "Constraint 2 must be satisfied"
        assert sum(coefs[2][j]*solver.Value(vars[j]) for j in range(num_vars)) <= random_numbers[2], \
        "Constraint 3 must be satisfied"
        assert sum(coefs[3][j]*solver.Value(vars[j]) for j in range(num_vars)) >= random_numbers[3], \
        "Constraint 4 must be satisfied"        
