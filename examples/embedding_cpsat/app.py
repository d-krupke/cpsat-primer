"""
This is an example of how to embed the CP-SAT solver in a Streamlit app.
We are using Python's multiprocessing module to run the solver in a separate process.
This way, the Streamlit app remains responsive while the solver is running.
By using pipes and signals, we can not only exchange the data between the processes
but also interrupt the solver process in a clean way.

Run this app with:
```
streamlit run app.py
```
"""

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

# Add a generate and run button and an interrupt button side by side
generate_button, interrupt_button = st.columns(2)
generate_button = generate_button.button("Generate and solve random geometric graph")
interrupt_button = interrupt_button.button("Abort")

plot_placeholder = st.empty()
if "plot" not in st.session_state:
    st.session_state.plot = None
plot_placeholder.pyplot(st.session_state.plot)

lb_ub_placeholder = st.empty()
if "lb_ub" not in st.session_state:
    st.session_state.lb_ub = ""
lb_ub_placeholder.markdown(st.session_state.lb_ub)

log_placeholder = st.empty()
if "log_text" not in st.session_state:
    st.session_state.log_text = ""
log_placeholder.text(st.session_state.log_text)

if generate_button:
    graph, points = generate_random_geometric_graph(int(n_points))

    ax = plot_instance(points)
    plot_placeholder.pyplot(ax.figure)
    solver_process = TspSolverProcess(graph)
    solver_process.start()

    st.session_state.log_text = ""
    st.session_state.plot = None

    while True:
        data = generate_data()

        solution = solver_process.get_solution()
        if solution is not None:
            ax = plot_solution(points, solution)
            st.session_state.plot = ax.figure
            plot_placeholder.pyplot(st.session_state.plot)
            plt.close(ax.figure)

        lb, ub = solver_process.get_bounds()
        st.session_state.lb_ub = f"**Lower bound: {lb}, Upper bound: {ub}**"
        lb_ub_placeholder.markdown(st.session_state.lb_ub)

        logs = solver_process.get_log()
        if logs:
            st.session_state.log_text += "\n".join(logs)+"\n"
        log_placeholder.text(st.session_state.log_text)

        if not solver_process.is_running() or interrupt_button:
            if interrupt_button:
                solver_process.interrupt()
                st.write("Solution process interrupted.")
            else:
                print("Solver finished.")
            break
        time.sleep(0.1)
