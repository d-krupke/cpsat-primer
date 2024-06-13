"""
This is just a simple main that also shows how to use the solver process.
For those not comfortable with the Streamlit app, this is a good starting point.
"""

from solver_process import TspSolverProcess
from tsp_solver import generate_random_geometric_graph
import time

if __name__ == "__main__":
    graph, points = generate_random_geometric_graph(200)
    tsp_solver = TspSolverProcess(graph)
    tsp_solver.start()
    for i in range(20):
        lb, ub = tsp_solver.get_bounds()
        print(f"Lower bound: {lb}, Upper bound: {ub}")
        solution = tsp_solver.get_solution()
        if solution is not None:
            print("Incumbent solution: ", solution)
        if not tsp_solver.is_running():
            break
        time.sleep(1)
    tsp_solver.interrupt()
    solution = tsp_solver.get_solution()
    if solution is not None:
        print("Solution found: ", solution)
    else:
        print("No solution found.")
