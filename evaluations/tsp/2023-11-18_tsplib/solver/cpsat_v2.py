"""
This file provides a simple implementation of a TSP solver using CP-SAT and the Dantzig-Fulkerson-Johnson formulation.
Because the CP-SAT solver does not support lazy constraints, we have to run the solver multiple times.
"""

import networkx as nx
from ortools.sat.python import cp_model
import typing
import time
import logging


class _EdgeVars:
    def __init__(self, model: cp_model.CpModel, graph: nx.Graph) -> None:
        self._model = model
        self._graph = graph
        self._vars = {
            (u, v): model.new_bool_var(f"edge_{u}_{v}") for u, v in graph.edges
        }

    def x(self, v, w):
        if (v, w) in self._vars:
            return self._vars[v, w]
        return self._vars[w, v]

    def outgoing_edges(self, vertices):
        for (v, w), x in self._vars.items():
            if v in vertices and w not in vertices:
                yield (v, w), x
            elif w in vertices and v not in vertices:
                yield (w, v), x

    def incident_edges(self, v):
        for n in self._graph.neighbors(v):
            yield (v, n), self.x(v, n)

    def __iter__(self):
        return iter(self._vars.items())

    def as_graph(self, value_f):
        """
        Return the current solution as a graph.
        """
        used_edges = [vw for vw, x in self if value_f(x)]
        return nx.Graph(used_edges)


class _SubtourCallback(cp_model.CpSolverSolutionCallback):
    """
    Callback to detect subtours during the search.
    """

    def __init__(self, edge_vars: _EdgeVars, early_abort=False):
        """
        edge_vars: a dictionary mapping edges to variables.
        early_abort: if True, the search will be aborted as soon as a subtour is detected. This is closer to lazy constraints, but may waste the previous work.
        """
        super().__init__()
        self.edge_vars = edge_vars
        self.subtours = []
        self.early_abort = early_abort
        self.best_solution = None
        self.best_objective = float("inf")

    def on_solution_callback(self):
        solution = self.edge_vars.as_graph(lambda x: self.Value(x))
        connected_components = list(nx.connected_components(solution))
        if len(connected_components) > 1:
            self.subtours += connected_components
            if self.early_abort:
                self.stop_search()
        else:
            # update best solution
            obj = self.objective_value
            if obj < self.best_objective:
                self.best_objective = obj
                self.best_solution = solution

    def has_subtours(self):
        return len(self.subtours) > 0

    def reset(self):
        self.subtours = []


class CpSatTspSolverDantzig:
    def __init__(
        self,
        G: nx.Graph,
        logger: typing.Optional[logging.Logger] = None,
        early_abort=False,
    ):
        self.logger = logger if logger else logging.getLogger("CpSatTspSolverV1")
        self.logger.info("Building model.")
        self.graph = G
        self._model = cp_model.CpModel()
        self._edge_vars = _EdgeVars(self._model, G)
        self.early_abort = early_abort
        self.best_bound = 0

        for v in G.nodes:
            self._model.add(sum(x for _, x in self._edge_vars.incident_edges(v)) == 2)

        # Objective
        self._model.minimize(
            sum(x * G[u][v]["weight"] for (u, v), x in self._edge_vars)
        )
        self.logger.info("Model built.")

    def solve(
        self, time_limit: float, opt_tol: float = 0.001
    ) -> typing.Tuple[float, float]:
        """
        Solve the model and return the objective value and the lower bound.
        """
        solver = cp_model.CpSolver()

        start_time = time.time()

        def remaining_time():
            return time_limit - (time.time() - start_time)

        solver.parameters.max_time_in_seconds = remaining_time()
        callback = _SubtourCallback(self._edge_vars, early_abort=self.early_abort)
        solver.parameters.log_search_progress = True
        solver.parameters.relative_gap_limit = opt_tol
        solver.log_callback = lambda s: self.logger.info(s)
        status = solver.solve(self._model, callback)
        self.best_bound = max(self.best_bound, solver.best_objective_bound)

        # The following part is more complex. Here we repeatedly add constraints
        # and solve the model again, until we find a solution without subtours.
        # TODO: Use the previous solutions as hints for the next solve.
        while callback.has_subtours():
            subtours = callback.subtours
            if len(subtours) > 1:
                for comp in subtours:
                    outgoing_edges = sum(
                        x for _, x in self._edge_vars.outgoing_edges(comp)
                    )
                    self._model.add(outgoing_edges >= 2)
                callback.reset()

                tour_cost = sum(
                    x * self.graph[u][v]["weight"] for (u, v), x in self._edge_vars
                )
                self._model.add(
                    tour_cost >= int(self.best_bound)
                )  # help with lower bound
                solver.parameters.max_time_in_seconds = remaining_time()
                if remaining_time() <= 0:
                    # Time limit reached without
                    return callback.best_objective, self.best_bound
                status = solver.solve(self._model, callback)
                self.best_bound = max(self.best_bound, solver.best_objective_bound)
            else:
                break
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.solution = callback.best_solution
            return callback.best_objective, self.best_bound
        # Check that the only reason for stopping before optimality is the time limit.
        assert status == cp_model.UNKNOWN, f"Unexpected status {status}"
        return callback.best_objective, self.best_bound
