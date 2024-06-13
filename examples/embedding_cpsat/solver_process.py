import multiprocessing
import signal
import os
from ortools.sat.python import cp_model

from typing import Dict, Optional, Callable

from tsp_solver import TspSolver, Tour, WeightedDirectedGraph


class InterprocessCallback(cp_model.CpSolverSolutionCallback):
    """
    This callback makes it easy to communicate the solver's progress
    to the main process. It sends the current solution to the main process
    through a pipe.
    """

    def __init__(
        self,
        shared_ub_value,
        shared_lb_value,
        solution_pipe,
        solution_extractor: Callable[[Callable[[cp_model.LinearExprT], int]], Dict],
    ):
        super().__init__()
        self.shared_ub_value = shared_ub_value
        self.shared_lb_value = shared_lb_value
        self.solution_data_pipe = solution_pipe
        self.solution_extractor = solution_extractor

    def on_solution_callback(self):
        self.shared_ub_value.value = self.ObjectiveValue()
        self.shared_lb_value.value = self.BestObjectiveBound()

        def get_value(x):
            return self.Value(x)

        self.solution_data_pipe.send(self.solution_extractor(get_value))


def _entry_point_solver_process(
    graph, max_time, lower_bound, upper_bound, log_conn, solution_conn
):
    """
    This function is the entry point for the optimization process.
    It is created via multiprocessing.Process and runs the TspSolver
    on the given graph. It communicates the progress and the solution
    back to the main process through the given pipes.
    """
    # === Signal handling ===
    # The interrupt signal is only directed to CP-SAT to interrupt the solver.
    # Ignore the signal while in the Python code, as we don't want to interrupt
    # the process itself.
    def signal_handler(sig, frame):
        # This function is called when the process receives a SIGINT signal
        # We actually don't want to do anything in this case, as we only
        # use the signal to interrupt CP-SAT. In case of bad timing, we
        # don't want to kill the process.
        pass

    signal.signal(signal.SIGINT, signal_handler)
    solver = TspSolver(graph)

    # === Logging ===
    # We need to activate the logging to send the log messages back to the main process.
    # We don't want to log to stdout, as this would be a waste of resources.
    # Instead, we log to the pipe, such that the log could be utilized in the main process.
    solver.solver.parameters.log_search_progress = True
    solver.solver.parameters.log_to_stdout = False
    solver.solver.log_callback = lambda msg: log_conn.send([str(msg)])

    # === Best bound callback ===
    # Update the shared lower bound value via a callback
    def update_lower_bound(x):
        lower_bound.value = x
    solver.solver.best_bound_callback = update_lower_bound

    # === Solution callback ===
    callback = InterprocessCallback(
        shared_ub_value=upper_bound,
        shared_lb_value=lower_bound,
        solution_pipe=solution_conn,
        solution_extractor=lambda get_value: solver.edge_vars.extract_tour(
            get_value
        ).model_dump(),
    )
    # === Solve ===
    status, solution = solver.solve(max_time, callback=callback)
    if solution is not None:
        solution_conn.send(solution.model_dump())  # Pydantic model's dict method
    log_conn.close()
    solution_conn.close()

class TspSolverProcess:
    """
    This class wraps the TspSolver class and runs it on a separate process.
    It provides methods to start, interrupt and get the solution of the solver.
    This way it can be used in a non-blocking way, e.g., in a GUI application.
    """

    def __init__(self, graph: WeightedDirectedGraph, max_time: float = 600.0):
        self.graph = graph
        self.max_time = max_time
        self.status = None
        self._shared_lb_value = multiprocessing.Value("d", float("-inf"))
        self._shared_ub_value = multiprocessing.Value("d", float("inf"))
        self._log_pipe = multiprocessing.Pipe(duplex=True)
        self._solution_pipe = multiprocessing.Pipe(duplex=True)
        self.process = multiprocessing.Process(
            target=_entry_point_solver_process,
            args=(
                self.graph,
                self.max_time,
                self._shared_lb_value,
                self._shared_ub_value,
                self._log_pipe[1],
                self._solution_pipe[1],
            ),
        )
        self._solution = None



    def start(self):
        """
        Starts the optimization process.
        """
        self.process.start()

    def interrupt(self):
        """
        Will interrupt the optimization process.
        """
        if self.process.pid is not None and self.process.is_alive():
            os.kill(self.process.pid, signal.SIGINT)

    def is_running(self):
        """
        Returns true if the optimization process is still running.
        """
        return self.process.is_alive()

    def get_solution(self) -> Optional[Tour]:
        """
        Return the latest solution found by the solver. If no solution is found,
        return None.
        """
        solution_data = None
        while self._solution_pipe[0].poll():
            # Only the latest solution is kept
            solution_data = self._solution_pipe[0].recv()
        if solution_data is not None:
            self._solution = Tour(**solution_data)
        return self._solution

    def get_bounds(self):
        return self._shared_lb_value.value, self._shared_ub_value.value

    def get_log(self) -> list[str]:
        """
        Returns the latest log entries from the solver.
        It does not return all log entries, only the latest ones, which
        have been written to the log pipe since the last call of this method.
        """
        logs = []
        while self._log_pipe[0].poll():
            new_log_entries = self._log_pipe[0].recv()
            logs += new_log_entries
        return logs
    
    def __del__(self):
        # Clean up the process when the object is deleted
        if self.process.is_alive():
            self.interrupt()
            self.process.join()
            self.process.close()
