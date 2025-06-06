from instance_schema import FacilityLocationInstance
from ortools.sat.python import cp_model


def solve1(
    instance: FacilityLocationInstance, time_limit: float
) -> tuple[list[int], int, int]:
    """
    Solve the facility location problem with a greedy heuristic.
    :param m: serving cost matrix
    :param f: opening cost vector
    :param k: number of facilities to open
    :return: tuple of selected facilities and total cost
    """

    assert instance.is_integral, "The instance must be integral for this solver."

    model = cp_model.CpModel()
    # x[j] = 1 if facility j is open
    x = [model.NewBoolVar(f"x[{j}]") for j in range(instance.num_facilities)]
    # y[i][j] = 1 if client i is served by facility j
    y = [
        [model.NewBoolVar(f"y[{i}][{j}]") for j in range(instance.num_facilities)]
        for i in range(instance.num_clients)
    ]
    # Each client must be served by exactly one facility
    for i in range(instance.num_clients):
        model.Add(sum(y[i][j] for j in range(instance.num_facilities)) == 1)
    # Each facility can serve clients only if it is open
    for i in range(instance.num_clients):
        for j in range(instance.num_facilities):
            model.Add(y[i][j] <= x[j])
    # Add objective function
    model.Minimize(
        sum(instance.opening_cost[j] * x[j] for j in range(instance.num_facilities))
        + sum(
            instance.get_allocation_cost(client=i, facility=j) * y[i][j]
            for i in range(instance.num_clients)
            for j in range(instance.num_facilities)
        )
    )
    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = time_limit
    solver.Solve(model)
    # Extract the solution
    selected_facilities = [
        j for j in range(instance.num_facilities) if solver.Value(x[j]) == 1
    ]
    total_cost = solver.ObjectiveValue()
    return selected_facilities, int(total_cost), int(solver.best_objective_bound)


def solve2(
    instance: FacilityLocationInstance, time_limit: float
) -> tuple[list[int], int, int]:
    """
    Solve the facility location problem with a greedy heuristic.
    :param m: serving cost matrix
    :param f: opening cost vector
    :param k: number of facilities to open
    :return: tuple of selected facilities and total cost
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    # x[j] = 1 if facility j is open
    x = [model.NewBoolVar(f"x[{j}]") for j in instance.iter_facilities()]
    # y[i][j] = 1 if client i is served by facility j
    y = [
        [model.NewBoolVar(f"y[{i}][{j}]") for j in instance.iter_facilities()]
        for i in instance.iter_clients()
    ]
    # Each client must be served by exactly one facility
    for i in instance.iter_clients():
        model.Add(sum(y[i][j] for j in instance.iter_facilities()) == 1)
    # Each facility can serve clients only if it is open
    for j in instance.iter_facilities():
        model.Add(
            sum(y[i][j] for i in instance.iter_clients()) <= x[j] * instance.num_clients
        )

    # Add objective function
    model.Minimize(
        sum(instance.opening_cost[j] * x[j] for j in instance.iter_facilities())
        + sum(
            instance.get_allocation_cost(client=i, facility=j) * y[i][j]
            for i in instance.iter_clients()
            for j in instance.iter_facilities()
        )
    )
    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = time_limit
    solver.Solve(model)
    # Extract the solution
    selected_facilities = [
        j for j in instance.iter_facilities() if solver.Value(x[j]) == 1
    ]
    total_cost = solver.ObjectiveValue()
    return selected_facilities, int(total_cost), int(solver.best_objective_bound)


def solve3(
    instance: FacilityLocationInstance, time_limit: float
) -> tuple[list[int], int, int]:
    """
    Solve the facility location problem with a greedy heuristic.
    :param m: serving cost matrix
    :param f: opening cost vector
    :param k: number of facilities to open
    :return: tuple of selected facilities and total cost
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    # x[j] = 1 if facility j is open
    x = [model.NewBoolVar(f"x[{j}]") for j in instance.iter_facilities()]
    # y[i][j] = 1 if client i is served by facility j
    y = [
        [model.NewBoolVar(f"y[{i}][{j}]") for j in instance.iter_facilities()]
        for i in instance.iter_clients()
    ]
    # Each client must be served by exactly one facility
    for i in instance.iter_clients():
        model.Add(sum(y[i][j] for j in instance.iter_facilities()) == 1)
    # Each facility can serve clients only if it is open
    for j in instance.iter_facilities():
        model.Add(sum(y[i][j] for i in instance.iter_clients()) == 0).OnlyEnforceIf(
            ~x[j]
        )

    # Add objective function
    model.Minimize(
        sum(instance.opening_cost[j] * x[j] for j in instance.iter_facilities())
        + sum(
            instance.get_allocation_cost(client=i, facility=j) * y[i][j]
            for i in instance.iter_clients()
            for j in instance.iter_facilities()
        )
    )
    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = time_limit
    solver.Solve(model)
    # Extract the solution
    selected_facilities = [
        j for j in instance.iter_facilities() if solver.Value(x[j]) == 1
    ]
    total_cost = solver.ObjectiveValue()
    return selected_facilities, int(total_cost), int(solver.best_objective_bound)
