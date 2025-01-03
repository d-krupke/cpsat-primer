import multiprocessing
import signal
import os
from ortools.sat.python import cp_model
from typing import Dict, Optional, Callable
from tsp_solver import TspSolver, Tour, WeightedDirectedGraph


class InterprocessCallback(cp_model.CpSolverSolutionCallback):
    """
    This callback facilitates communication of the solver's progress
    to the main process by sending the current solution through a pipe.
    """

    def __init__(
        self,
        shared_ub_value,
        shared_lb_value,
        solution_pipe,
        solution_extractor: Callable[[Callable[[cp_model.LinearExprT], int]], Dict],
    ):
        """
        Args:
            shared_ub_value: A shared value for the upper bound.
            shared_lb_value: A shared value for the lower bound.
            solution_pipe: A pipe for sending the solution data.
            solution_extractor: A function that extracts the solution data from the solver.
                                It takes a function that can get the value of a variable and
                                returns a dictionary of the solution data.
        """
        super().__init__()
        self.shared_objective_value = shared_ub_value
        self.shared_bound_value = shared_lb_value
        self.solution_data_pipe = solution_pipe
        self.solution_extractor = solution_extractor

    def on_solution_callback(self):
        """Called when the solver finds a new solution."""
        self.shared_objective_value.value = self.objective_value
        self.shared_bound_value.value = self.best_objective_bound

        def get_value(var):
            return self.Value(var)

        solution_data = self.solution_extractor(get_value)
        self.solution_data_pipe.send(solution_data)


def _entry_point_solver_process(
    graph, max_time, lower_bound, upper_bound, log_conn, solution_conn
):
    """
    Entry point for the optimization process. Runs the TspSolver on the given graph and
    communicates progress and solutions to the main process through pipes.
    """

    # Signal handler for SIGINT signal, no operation needed.
    # This is to prevent the solver from crashing when interrupted, as we are using
    # SIGINT to tell CP-SAT to stop the search.
    def signal_handler(sig, frame):
        pass

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize the solver
    solver = TspSolver(graph)

    # Set solver parameters for logging
    # We want it to log the search progress, but to use the callback and not printing stdout
    solver.solver.parameters.log_search_progress = True
    solver.solver.parameters.log_to_stdout = False

    # Define log callback to send log messages through the log pipe
    solver.solver.log_callback = lambda msg: log_conn.send([str(msg)])

    # Define callback to update the shared lower bound value
    def update_lower_bound(value):
        lower_bound.value = value

    solver.solver.best_bound_callback = update_lower_bound

    # Create a callback instance to communicate solver progress
    callback = InterprocessCallback(
        shared_ub_value=upper_bound,
        shared_lb_value=lower_bound,
        solution_pipe=solution_conn,
        solution_extractor=lambda get_value: solver.edge_vars.extract_tour(
            get_value
        ).model_dump(),
    )

    try:
        # Solve the problem with the specified maximum time and callback
        status, solution = solver.solve(max_time, callback=callback)

        # If a solution is found, send the final solution through the solution pipe
        if solution is not None:
            solution_conn.send(solution.model_dump())
    finally:
        # Close the communication pipes
        log_conn.close()
        solution_conn.close()


class TspSolverProcess:
    """
    Wrapper for the TspSolver class that runs it in a separate process.
    Provides methods to start, interrupt, and retrieve the solution in a non-blocking manner.
    """

    def __init__(self, graph: WeightedDirectedGraph, max_time: float = 600.0):
        self.graph = graph
        self.max_time = max_time
        self.status = None
        self._shared_bound_value = multiprocessing.Value("d", float("-inf"))
        self._shared_objective_value = multiprocessing.Value("d", float("inf"))
        self._log_pipe = multiprocessing.Pipe(duplex=True)
        self._solution_pipe = multiprocessing.Pipe(duplex=True)
        self.process = multiprocessing.Process(
            target=_entry_point_solver_process,
            args=(
                self.graph,
                self.max_time,
                self._shared_bound_value,
                self._shared_objective_value,
                self._log_pipe[1],
                self._solution_pipe[1],
            ),
        )
        self._solution = None

    def start(self):
        """Starts the optimization process."""
        self.process.start()

    def interrupt(self):
        """Interrupts the optimization process."""
        self.process.join(timeout=1)
        if self.process.pid and self.process.is_alive():
            os.kill(self.process.pid, signal.SIGINT)

    def is_running(self):
        """Returns True if the optimization process is still running."""
        return self.process.is_alive()

    def get_solution(self) -> Optional[Tour]:
        """Returns the latest solution found by the solver, or None if no solution is found."""
        solution_data = None
        while self._solution_pipe[0].poll():
            solution_data = self._solution_pipe[0].recv()
        if solution_data is not None:
            self._solution = Tour(**solution_data)
        return self._solution

    def get_current_objective_value(self):
        """Returns the current objective value."""
        return self._shared_objective_value.value

    def get_current_bound(self):
        """Returns the current lower bound."""
        return self._shared_bound_value.value

    def get_log(self) -> list[str]:
        """Returns the latest log entries from the solver."""
        logs = []
        while self._log_pipe[0].poll():
            logs.extend(self._log_pipe[0].recv())
        return logs

    def __del__(self):
        """Cleans up the process when the object is deleted."""
        if self.process.is_alive():
            self.interrupt()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
            self.process.close()
