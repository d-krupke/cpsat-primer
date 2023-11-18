"""
This file provides a simple implementation of a TSP solver using Gurobi.

Check out the book "In Pursuit of the Traveling Salesman" by William J Cook for a great introduction to the TSP.
Check out the educational material on the Gurobi website (www.gurobi.com) for a great introduction to Gurobi and MIP.
"""

import gurobipy as gp
import typing
import networkx as nx
import logging


class _EdgeVariables:
    def __init__(self, G: nx.Graph, model: gp.Model):
        self._graph = G
        self._model = model
        self._vars = {
            (u, v): model.addVar(vtype=gp.GRB.BINARY, name=f"edge_{u}_{v}")
            for u, v in G.edges
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

    def as_graph(self, in_callback: bool = False):
        """
        Return the current solution as a graph.
        """
        if in_callback:
            used_edges = [vw for vw, x in self if self._model.cbGetSolution(x) > 0.5]
        else:
            used_edges = [vw for vw, x in self if x.X > 0.5]
        return nx.Graph(used_edges)


class GurobiTspSolver:
    def __init__(self, G: nx.Graph, logger: typing.Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger("GurobiTspSolver")
        self.logger.info("Building model.")
        self.graph = G
        self._model = gp.Model()
        self._vars = _EdgeVariables(G, self._model)
        self._build_model()
        self.solution = None
        self.logger.info("Model built.")

    def _build_model(self):
        # Constraints
        # Every nodes has two incident edges.
        for v in self.graph.nodes:
            self._model.addConstr(sum(x for _, x in self._vars.incident_edges(v)) == 2)

        # Objective
        def dist(u, v):
            return self.graph.edges[u, v]["weight"]

        tour_cost = sum(dist(v, w) * x for (v, w), x in self._vars)
        self._model.setObjective(tour_cost, gp.GRB.MINIMIZE)

    def solve(
        self, time_limit: float, opt_tol: float = 0.001
    ) -> typing.Tuple[float, float]:
        """
        Solve the model and return the objective value and the lower bound.
        """
        self._model.Params.LogToConsole = 1
        self._model.Params.TimeLimit = time_limit
        self._model.Params.lazyConstraints = 1
        self._model.Params.MIPGap = (
            opt_tol  # https://www.gurobi.com/documentation/10.0/refman/mipgap.html
        )

        def gurobi_subtour_callback(model, where):
            if where == gp.GRB.Callback.MESSAGE:
                msg = model.cbGet(gp.GRB.Callback.MSG_STRING)
                self.logger.info(msg)
            if where == gp.GRB.Callback.MIPSOL:
                connected_components = list(
                    nx.connected_components(self._vars.as_graph(in_callback=True))
                )
                if len(connected_components) > 1:
                    for comp in connected_components:
                        model.cbLazy(
                            sum(x for _, x in self._vars.outgoing_edges(comp)) >= 2
                        )

        self._model.optimize(gurobi_subtour_callback)
        # Should only terminate on optimal solution or time limit.
        assert self._model.status in (
            gp.GRB.OPTIMAL,
            gp.GRB.TIME_LIMIT,
        ), f"Unexpected status: {self._model.status}"
        if self._model.SolCount > 0:
            self.solution = self._vars.as_graph()
        return self._model.objVal, self._model.ObjBound
