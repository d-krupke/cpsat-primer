import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from tsp_solver import generate_random_geometric_graph
from solver_process import TspSolverProcess


# Function to create a new data set
def generate_data():
    return np.random.randn(100)


# Title of the app
st.title("TSP Solving with CP-SAT")


def plot_instance(points):
    """
    Plot the instance of the TSP problem.
    """
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points), c="red", s=10)
    return ax


def plot_solution(points, tour):
    """
    Plot the solution to the TSP problem.
    """
    ax = plot_instance(points)
    # tour is a list of indices of the points in the order they are visited
    xs = [points[i][0] for i in tour.sequence]
    ys = [points[i][1] for i in tour.sequence]
    xs.append(xs[0])
    ys.append(ys[0])
    ax.plot(xs, ys, c="blue")
    return ax


# add a field for the number of points
n_points = st.number_input("Number of points", min_value=2, value=10)

generate_button = st.button("Generate and solve random geometric graph")

plot_placeholder = st.empty()
lb_ub_placeholder = st.empty()

log_placeholder = st.empty()
log_text = ""

if generate_button:
    graph, points = generate_random_geometric_graph(int(n_points))

    ax = plot_instance(points)
    plot_placeholder.pyplot(ax.figure)
    solver_process = TspSolverProcess(graph)
    solver_process.start()

    while True:
        data = generate_data()

        solution = solver_process.get_solution()
        if solution is not None:
            ax = plot_solution(points, solution)
            plot_placeholder.pyplot(ax.figure)
            plt.close(ax.figure)
        lb, ub = solver_process.get_bounds()
        lb_ub_placeholder.text(f"Lower bound: {lb}, Upper bound: {ub}")
        logs = solver_process.get_log()
        log_text += "\n".join(logs)
        log_placeholder.text(log_text)

        if not solver_process.is_running():
            print("Solver finished.")
            break
        time.sleep(1)
