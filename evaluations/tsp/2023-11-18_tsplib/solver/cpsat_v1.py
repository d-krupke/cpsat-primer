"""
This file provides a simple TSP implementation using CP-SAT's add_circuit constraint.
"""

import networkx as nx
from ortools.sat.python import cp_model
import typing
import logging


class CpSatTspSolverV1:
    def __init__(self, G: nx.Graph, logger: typing.Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger("CpSatTspSolverV1")
        self.logger.info("Building model.")
        self.graph = G
        self._model = cp_model.CpModel()

        # Variables
        edge_vars = dict()
        for u, v in G.edges:
            edge_vars[u, v] = self._model.new_bool_var(f"edge_{u}_{v}")
            edge_vars[v, u] = self._model.new_bool_var(f"edge_{v}_{u}")

        # Constraints
        # Because the nodes in the graph a indices 0, 1, ..., n-1, we can use the
        # indices directly in the constraints. Otherwise, we would have to use
        # a mapping from the nodes to indices.
        circuit = [(u, v, x) for (u, v), x in edge_vars.items()]
        self._model.add_circuit(circuit)

        # Objective
        self._model.minimize(
            sum(x * G[u][v]["weight"] for (u, v), x in edge_vars.items())
        )
        self.logger.info("Model built.")

    def solve(
        self, time_limit: float, opt_tol: float = 0.001
    ) -> typing.Tuple[float, float]:
        """
        Solve the model and return the objective value and the lower bound.
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = True
        solver.log_callback = lambda s: self.logger.info(s)
        solver.parameters.relative_gap_limit = opt_tol
        status = solver.solve(self._model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return solver.objective_value, solver.best_objective_bound
        # Check that the only reason for stopping before optimality is the time limit.
        assert status in (
            cp_model.FEASIBLE,
            cp_model.UNKNOWN,
        ), f"Unexpected status {status}"
        return float("inf"), 0.0
