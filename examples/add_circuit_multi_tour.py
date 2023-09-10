"""
This example shows how to use the AddCircuit constraint to model a Multi TSP.
In the Multi TSP, we want to find k tours that visit all nodes exactly once,
of minimum total cost (sum of circle lengths).
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
    random.seed()
    graph = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                graph[(u,v)] = random.randint(0, 100)
    return graph

if __name__ == '__main__':
    # Weighted, directed graph as instance
    # (source, destination) -> cost
    num_nodes = 100
    dgraph: Dict[Tuple[int, int], int] = generate_random_graph(num_nodes)
    num_tours = 2  

    model = cp_model.CpModel()
    # Variables: Binary decision variables for the edges in each tour
    edge_vars = [{ 
        (u,v): model.NewBoolVar(f"e_{u}_{v}") for (u,v) in dgraph.keys()
    } for _ in range(num_tours)]
    # Variables: Binary decision variables for the vertices in each tour
    vertex_vars = [{
        u: model.NewBoolVar(f"v_{u}") for u in range(num_nodes)
    } for _ in range(num_tours)]
    # Constraints: Add Circuit constraint
    # We need to tell CP-SAT which variable corresponds to which edge.
    # This is done by passing a list of tuples (u,v,var) to AddCircuit.
    for i in range(num_tours):
        circuit = [(u, v, var)  # (source, destination, variable)
                    for (u,v),var in edge_vars[i].items()]
        # Add skipping variables to each circuit because not all vertices are essential. CP-SAT will detect them by
        # v==v and not force v to be in the circuit, if the associated literal is true. Not() such that
        circuit += [(v,v, var.Not())  # var==True <=> var.Not() == False(literal is false) <=> v in circuit
                        for v,var in vertex_vars[i].items()]
        model.AddCircuit(circuit)

    # Constraints: Add constraint that each vertex is in exactly one tour
    for v in range(num_nodes):
        model.Add(sum(vertex_vars[i][v] for i in range(num_tours)) == 1)

    # Objective: minimize the total cost of edges
    obj = sum(dgraph[(u,v)]*var for i in range(num_tours) for (u,v),var in edge_vars[i].items())
    model.Minimize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    # Print solution
    if status == cp_model.OPTIMAL:
        tours = [[(u,v) for (u,v),var in edge_vars[i].items() if solver.Value(var)] for i in range(num_tours)]
        print("Optimal tours are: ", sorted(tours))
        print("The cost of the tours are: ", solver.ObjectiveValue())
    elif status == cp_model.FEASIBLE:
        tours = [[(u,v) for (u,v),var in edge_vars[i].items() if solver.Value(var)] for i in range(num_tours)]
        print("Optimal tours are: ", sorted(tours))
        print("The cost of the tours are: ", solver.ObjectiveValue())
        print("The lower bound of the tour is: ", solver.BestObjectiveBound())
    else:
        print("No solution found.")
