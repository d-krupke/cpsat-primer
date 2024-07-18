from ortools.sat.python import cp_model


def test_circuit():
    # Weighted, directed graph as instance
    # (source, destination) -> cost
    dgraph = {
        (0, 1): 13,
        (1, 0): 17,
        (1, 2): 16,
        (2, 1): 19,
        (0, 2): 22,
        (2, 0): 14,
        (3, 0): 15,
        (3, 1): 28,
        (3, 2): 25,
        (0, 3): 24,
        (1, 3): 11,
        (2, 3): 27,
    }
    model = cp_model.CpModel()
    # Variables: Binary decision variables for the edges
    edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph.keys()}
    # Constraints: Add Circuit constraint
    # We need to tell CP-SAT which variable corresponds to which edge.
    # This is done by passing a list of tuples (u,v,var) to AddCircuit.
    circuit = [
        (u, v, var)
        for (u, v), var in edge_vars.items()  # (source, destination, variable)
    ]
    model.add_circuit(circuit)

    # Objective: minimize the total cost of edges
    obj = sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items())
    model.Minimize(obj)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    tour = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x)]
    assert len(tour) == 4


def test_shortest_path():
    # Define a weighted, directed graph (source, destination) -> cost
    dgraph = {
        (0, 1): 13,
        (1, 0): 17,
        (1, 2): 16,
        (2, 1): 19,
        (0, 2): 22,
        (2, 0): 14,
        (3, 1): 28,
        (3, 2): 25,
        (0, 3): 30,
        (1, 3): 11,
        (2, 3): 27,
    }

    source_vertex = 0
    target_vertex = 3

    # Add zero-cost loop edges for all vertices that are not the source or target
    # This will allow CP-SAT to skip these vertices.
    for v in [1, 2]:
        dgraph[(v, v)] = 0

    # Create CP-SAT model
    model = cp_model.CpModel()

    # Create binary decision variables for each edge in the graph
    edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph.keys()}

    # Add a circuit constraint to the model
    circuit = [
        (u, v, var)
        for (u, v), var in edge_vars.items()  # (source, destination, variable)
    ]
    # Add enforced pseudo-edge from target to source to close the path
    circuit.append((target_vertex, source_vertex, 1))

    model.add_circuit(circuit)

    # Objective: minimize the total cost of edges
    obj = sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items())
    model.minimize(obj)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    path = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x) and u != v]
    assert len(path) == 2
