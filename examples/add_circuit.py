"""
This example shows how to use the AddCircuit constraint to model a TSP.
It can solve instances of reasonable size to optimality.
The performance may depend on the weights of the edges, with
euclidean distances probably being the easiest instances.
Random weights may be harder to solve, as they do not obey the triangle
inequality, which changes the theoretical properties of the problem.
"""

from ortools.sat.python import cp_model
from typing import Dict, Tuple

def generate_random_graph(n, seed=None):
    """Generate a random weighted graph with n nodes and n*(n-1) edges."""
    import random
    random.seed(seed)
    graph = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                graph[(u,v)] = random.randint(0, 100)
    return graph

if __name__ == '__main__':
    # Weighted, directed graph as instance
    # (source, destination) -> cost
    dgraph: Dict[Tuple[int, int], int] = generate_random_graph(150)
    
    model = cp_model.CpModel()
    # Variables: Binary decision variables for the edges
    edge_vars = { 
        (u,v): model.NewBoolVar(f"e_{u}_{v}") for (u,v) in dgraph.keys()
    }
    # Constraints: Add Circuit constraint
    # We need to tell CP-SAT which variable corresponds to which edge.
    # This is done by passing a list of tuples (u,v,var) to AddCircuit.
    circuit = [(u, v, var)  # (source, destination, variable)
                for (u,v),var in edge_vars.items()]
    model.AddCircuit(circuit)

    # Objective: minimize the total cost of edges
    obj = sum(dgraph[(u,v)]*var for (u,v),var  in edge_vars.items())
    model.Minimize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    # Print solution
    if status == cp_model.OPTIMAL:
        tour = [(u,v) for (u,v),var in edge_vars.items() if solver.Value(var)]
        print("Optimal tour is: ", sorted(tour))
        print("The cost of the tour is: ", solver.ObjectiveValue())
    elif status == cp_model.FEASIBLE:
        tour = [(u,v) for (u,v),var in edge_vars.items() if solver.Value(var)]
        print("Feasible tour is: ", sorted(tour))
        print("The cost of the tour is: ", solver.ObjectiveValue())
        print("The lower bound of the tour is: ", solver.BestObjectiveBound())
    else:
        print("No solution found.")
