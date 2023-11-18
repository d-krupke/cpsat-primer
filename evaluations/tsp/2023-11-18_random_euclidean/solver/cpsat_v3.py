"""
This file implements the MTZ formulation of the TSP using CP-SAT.
"""
import networkx as nx
from ortools.sat.python import cp_model
import typing
import logging


class CpSatTspSolverMtz:
    def __init__(self, G: nx.Graph, logger: typing.Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger("CpSatTspSolverV1")
        self.logger.info("Building model.")
        self.graph = G
        self._model = cp_model.CpModel()

        # Variables
        edge_vars = dict()
        for u, v in G.edges:
            edge_vars[u, v] = self._model.NewBoolVar(f"edge_{u}_{v}")
            edge_vars[v, u] = self._model.NewBoolVar(f"edge_{v}_{u}")

        depth_vars = dict()
        for u in G.nodes:
            depth_vars[u] = self._model.NewIntVar(
                0, G.number_of_nodes() - 1, f"depth_{u}"
            )

        # Constraints
        # Every node has exactly one incoming and one outgoing edge.
        for v in G.nodes:
            self._model.AddExactlyOne([edge_vars[u, v] for u in G.neighbors(v)])
            self._model.AddExactlyOne([edge_vars[v, u] for u in G.neighbors(v)])

        # Use depth variables to prohibit subtours.
        self._model.Add(depth_vars[0] == 0)  # fix the root node to depth 0
        for v, w in G.edges:
            if w != 0:  # The root node is special.
                self._model.Add(depth_vars[v] + 1 == depth_vars[w]).OnlyEnforceIf(
                    edge_vars[v, w]
                )
            if v != 0:
                self._model.Add(depth_vars[w] + 1 == depth_vars[v]).OnlyEnforceIf(
                    edge_vars[w, v]
                )

        # Objective
        self._model.Minimize(
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
        solver.parameters.relative_gap_limit = opt_tol
        solver.log_callback = lambda s: self.logger.info(s)
        status = solver.Solve(self._model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return solver.ObjectiveValue(), solver.BestObjectiveBound()
        # Check that the only reason for stopping before optimality is the time limit.
        assert status in (
            cp_model.FEASIBLE,
            cp_model.UNKNOWN,
        ), f"Unexpected status {status}"
        return float("inf"), 0.0
