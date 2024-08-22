"""
This example shows how to use the AddCircuit constraint to model the Budget TSP.
The Budget TSP is a variant of the TSP, where each edge has a cost and we want
to visit as many nodes as possible, while respecting a budget constraint on
the total cost of the tour.
It can solve instances of reasonable size to optimality.
The performance may depend on the weights of the edges, with
euclidean distances probably being the easiest instances.
Random weights may be harder to solve, as they do not obey the triangle
inequality, which changes the theoretical properties of the problem.
"""

from ortools.sat.python import cp_model
from typing import Dict, Tuple


def generate_random_graph(n, seed=None):
    """Generate a random graph with n nodes and n*(n-1) edges."""
    import random

    random.seed(seed)
    graph = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                graph[(u, v)] = random.randint(10, 100)
    return graph


if __name__ == "__main__":
    # Weighted, directed graph as instance
    # (source, destination) -> cost
    n = 100
    dgraph: Dict[Tuple[int, int], int] = generate_random_graph(n)
    budget: int = n * 10

    model = cp_model.CpModel()
    # Variables: Binary decision variables for the edges
    edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph.keys()}
    # Variables: Binary decision variables for the vertices
    vertex_vars = {u: model.new_bool_var(f"v_{u}") for u in range(n)}
    # Constraints: Add Circuit constraint
    # We need to tell CP-SAT which variable corresponds to which edge.
    # This is done by passing a list of tuples (u,v,var) to AddCircuit.
    circuit = [
        (u, v, var)  # (source, destination, variable)
        for (u, v), var in edge_vars.items()
    ]
    # Add skipping variables to the circuit. CP-SAT will detect them by
    # v==v and not force v to be in the circuit, if the variable is false.
    circuit += [
        (v, v, ~var)  # ~var such that var==True <=> v in circuit
        for v, var in vertex_vars.items()
    ]
    model.add_circuit(circuit)

    # Constraints: Budget constraint
    tour_cost = sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items())
    model.add(tour_cost <= budget)

    # Objective: Maximize the number of visited nodes
    obj = sum(vertex_vars.values())
    model.maximize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    status = solver.solve(model)

    # Print solution
    if status == cp_model.OPTIMAL:  # Found an optimal solution
        tour = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x)]
        print("Optimal tour is: ", sorted(tour))
        print("Number of vertices in the tour:", solver.objective_value)
        print("Cost of tour:", sum(dgraph[(u, v)] for (u, v) in tour))
    elif status == cp_model.FEASIBLE:  # Found a feasible solution
        tour = [(u, v) for (u, v), x in edge_vars.items() if solver.Value(x)]
        print("Feasible tour is: ", sorted(tour))
        print("Number of vertices in the tour:", solver.objective_value)
        print("The upper bound is: ", solver.best_objective_bound)
        print("Cost of tour:", sum(dgraph[(u, v)] for (u, v) in tour))
    else:  # No solution found
        print("No solution found.")
