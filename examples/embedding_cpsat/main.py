"""
This script demonstrates how to use the TSP solver process.
For those not comfortable with the Streamlit app, this script is a good starting point.
"""

from solver_process import TspSolverProcess
from tsp_solver import generate_random_geometric_graph
import time


def main():
    """
    Main function to demonstrate the TSP solver process.
    """
    # Generate a random geometric graph with 200 vertices
    graph, points = generate_random_geometric_graph(200)
    
    # Initialize and start the TSP solver process
    tsp_solver = TspSolverProcess(graph)
    tsp_solver.start()
    
    # Iterate and monitor the solver's progress
    for _ in range(20):
        lower_bound = tsp_solver.get_current_bound()
        upper_bound = tsp_solver.get_current_objective_value()
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        
        solution = tsp_solver.get_solution()
        if solution is not None:
            print("Incumbent solution: ", solution)
        
        if not tsp_solver.is_running():
            break
        
        time.sleep(1)
    
    # Interrupt the solver process and get the final solution
    tsp_solver.interrupt()
    solution = tsp_solver.get_solution()
    
    if solution is not None:
        print("Solution found: ", solution)
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
