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
    """Generate a weighted random graph with n nodes and n*(n-1) edges."""
    import random
    random.seed(seed)
    graph = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                graph[(u,v)] = random.randint(10, 100)
    return graph

if __name__ == '__main__':
    # Weighted, directed graph as instance
    # (source, destination) -> cost
    num_nodes = 100
    dgraph: Dict[Tuple[int, int], int] = generate_random_graph(num_nodes)
    budget: int = num_nodes*10
    
    model = cp_model.CpModel()
    # Variables: Binary decision variables for the edges
    edge_vars = { 
        (u,v): model.NewBoolVar(f"e_{u}_{v}") for (u,v) in dgraph.keys()
    }
    # Variables: Binary decision variables for the vertices
    vertex_vars = {
        u: model.NewBoolVar(f"v_{u}") for u in range(num_nodes)
    }
    # Constraints: Add Circuit constraint
    # We need to tell CP-SAT which variable corresponds to which edge.
    # This is done by passing a list of tuples (u,v,var) to AddCircuit.
    circuit = [(u, v, var)  # (source, destination, variable)
                for (u,v),var in edge_vars.items()]
    # Add skipping variables to the circuit because not all vertices are essential. CP-SAT will detect them by
    # v==v and not force v to be in the circuit, if the associated literal is true. Not() such that
    circuit += [(v,v, var.Not())  # var==True <=> var.Not() == False(literal is false) <=> v in circuit
                 for v,var in vertex_vars.items()]
    model.AddCircuit(circuit)

    # Constraints: Budget constraint
    tour_cost = sum(dgraph[(u,v)]*var for (u,v),var  in edge_vars.items())
    model.Add(tour_cost <= budget)

    # Objective: Maximize the number of visited nodes
    obj = sum(vertex_vars.values())
    model.Maximize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    # Print solution
    if status == cp_model.OPTIMAL:  # Found an optimal solution
        tour = [(u,v) for (u,v),var in edge_vars.items() if solver.Value(var)]
        print("Optimal tour is: ", sorted(tour))
        print("Number of vertices in the tour:", solver.ObjectiveValue())
        print("Cost of tour:", sum(dgraph[(u,v)] for (u,v) in tour))
    elif status == cp_model.FEASIBLE:  # Found a feasible solution
        tour = [(u,v) for (u,v),var in edge_vars.items() if solver.Value(var)]
        print("Feasible tour is: ", sorted(tour))
        print("Number of vertices in the tour:", solver.ObjectiveValue())
        print("The upper bound is: ", solver.BestObjectiveBound())
        print("Cost of tour:", sum(dgraph[(u,v)] for (u,v) in tour))
    else:  # No solution found
        print("No solution found.")
