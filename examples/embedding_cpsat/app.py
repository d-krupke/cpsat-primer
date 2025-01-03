"""
This script demonstrates how to embed the CP-SAT solver in a Streamlit app.
We use Python's multiprocessing module to run the solver in a separate process,
ensuring the Streamlit app remains responsive while the solver is running.
By using pipes and signals, we can exchange data between the processes
and interrupt the solver process in a clean way.

Run this app with:
```
streamlit run app.py
```

CAVEAT: This app allows to start many solver processes in parallel, which
can overload the system. For a production system, you would have to extract
the solver to a separate service which can queue and manage the requests.
"""

import streamlit as st
import matplotlib.pyplot as plt
import time
from tsp_solver import generate_random_geometric_graph
from solver_process import TspSolverProcess


def plot_instance(points):
    """
    Plot the instance of the TSP problem.

    Args:
        points (List[Tuple[int, int]]): The coordinates of the points.

    Returns:
        matplotlib.figure.Figure: The plot of the points.
    """
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points), c="red", s=10)
    ax.set_title("TSP Instance")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    return fig


def plot_solution(points, tour):
    """
    Plot the solution to the TSP problem.

    Args:
        points (List[Tuple[int, int]]): The coordinates of the points.
        tour (Tour): The tour representing the TSP solution.

    Returns:
        matplotlib.figure.Figure: The plot of the solution.
    """
    fig = plot_instance(points)
    ax = fig.axes[0]
    xs = [points[i][0] for i in tour.sequence]
    ys = [points[i][1] for i in tour.sequence]
    xs.append(xs[0])
    ys.append(ys[0])
    ax.plot(xs, ys, c="blue")
    ax.set_title("TSP Solution")
    return fig


def calculate_progress(lower_bound, upper_bound):
    """
    Calculate progress as a percentage based on lower and upper bounds.

    Args:
        lower_bound (float): The current lower bound of the solution.
        upper_bound (float): The current upper bound of the solution.

    Returns:
        float: Progress percentage.
    """
    if lower_bound == float("-inf") or upper_bound == float("inf"):
        return 0.0
    return max(0.0, min(1.0, (lower_bound / upper_bound) if upper_bound != 0 else 0.0))


# Title of the app
st.title("TSP Solving with CP-SAT")

st.markdown("""
    This app generates a random number of points and solves the [Traveling Salesman Problem (TSP)](https://simple.wikipedia.org/wiki/Travelling_salesman_problem) using the [CP-SAT solver](https://developers.google.com/optimization/cp?hl=fr).
""")

# Add a field for the number of points and generate/abort buttons with improved layout
st.sidebar.header("Configuration")
num_points = st.sidebar.number_input("Number of points", min_value=2, value=100)

generate_button = st.sidebar.button("Generate and run")
interrupt_button = st.sidebar.button("Abort")

# Reference
st.sidebar.markdown(
    """
    This app is part of a CP-SAT teaching series. It shows how to embed the CP-SAT solver in an app with multiprocessing.

    Learn more about solving hard combinatorial problems [with the CP-SAT primer.](https://github.com/d-krupke/cpsat-primer/tree/main?tab=readme-ov-file)
    """
)

plot_placeholder = st.empty()
if "plot" not in st.session_state:
    st.session_state.plot = None
if st.session_state.plot is not None:
    plot_placeholder.pyplot(st.session_state.plot)
else:
    # Display instructions
    st.info("Click 'Generate and run' on the sidebar to start the solver.")

lb_ub_placeholder = st.empty()
if "lb_ub" not in st.session_state:
    st.session_state.lb_ub = ""
lb_ub_placeholder.markdown(st.session_state.lb_ub)

progress_bar = st.progress(0)

log_placeholder = st.empty()
if "log_text" not in st.session_state:
    st.session_state.log_text = ""
log_placeholder.code(st.session_state.log_text, language="text")


if generate_button:
    graph, points = generate_random_geometric_graph(int(num_points))

    fig = plot_instance(points)
    plot_placeholder.pyplot(fig)
    solver_process = TspSolverProcess(graph)
    solver_process.start()

    st.session_state.log_text = ""
    st.session_state.plot = None

    while True:
        solution = solver_process.get_solution()
        if solution is not None:
            fig = plot_solution(points, solution)
            st.session_state.plot = fig
            plot_placeholder.pyplot(st.session_state.plot)
            plt.close(fig)

        lower_bound = solver_process.get_current_bound()
        upper_bound = solver_process.get_current_objective_value()
        st.session_state.lb_ub = (
            f"**Lower bound: {lower_bound}, Upper bound: {upper_bound}**"
        )
        lb_ub_placeholder.markdown(st.session_state.lb_ub)

        logs = solver_process.get_log()
        if logs:
            st.session_state.log_text += "\n".join(logs) + "\n"
        log_placeholder.code(st.session_state.log_text, language="text")

        progress = calculate_progress(lower_bound, upper_bound)
        progress_bar.progress(progress)

        if not solver_process.is_running() or interrupt_button:
            if interrupt_button:
                solver_process.interrupt()
                st.write("Solution process interrupted.")
            else:
                st.write("Solver finished.")
            break
        time.sleep(0.1)
