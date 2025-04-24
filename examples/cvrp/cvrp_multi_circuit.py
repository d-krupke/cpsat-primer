from typing import Hashable
import networkx as nx
from ortools.sat.python import cp_model


class CapacitatedMultiCircuit:
    """CVRP via CP-SAT multi-circuit constraint."""

    def __init__(
        self,
        graph: nx.Graph,
        depot: Hashable,
        capacity: int,
        demand_label: str = "demand",
        model: cp_model.CpModel | None = None,
    ):
        self.graph, self.depot = graph, depot
        self.model = model or cp_model.CpModel()
        self.capacity = capacity
        self.demand_label = demand_label

        # Vertex list with depot first
        self.vertices = [depot] + [v for v in graph.nodes() if v != depot]
        self.index = {v: i for i, v in enumerate(self.vertices)}

        # Boolean arc variables for both directions
        self.arc_vars = {
            (i, j): self.model.new_bool_var(f"arc_{i}_{j}")
            for u, v in graph.edges
            for i, j in ((self.index[u], self.index[v]), (self.index[v], self.index[u]))
        }
        arcs = [(i, j, var) for (i, j), var in self.arc_vars.items()]

        # Multi-circuit constraint
        self.model.add_multiple_circuit(arcs)

        # Capacity variables and constraints
        self.cap_vars = [
            self.model.new_int_var(0, capacity, f"cap_{i}")
            for i in range(len(self.vertices))
        ]
        for i, j, var in arcs:
            if j == 0:
                continue
            demand = graph.nodes[self.vertices[j]].get(demand_label, 0)
            self.model.add(
                self.cap_vars[j] >= self.cap_vars[i] + demand
            ).only_enforce_if(var)

    def is_arc_used(self, u, v) -> cp_model.BoolVarT:
        return self.arc_vars[(self.index[u], self.index[v])]

    def weight(self, label: str = "weight") -> cp_model.LinearExprT:
        return sum(
            var * self.graph[self.vertices[i]][self.vertices[j]][label]
            for (i, j), var in self.arc_vars.items()
        )

    def minimize_weight(self, label: str = "weight"):
        self.model.minimize(self.weight(label=label))

    def extract_tours(self, solver: cp_model.CpSolver) -> list[list]:
        # Build directed graph of selected arcs
        dg = nx.DiGraph(
            [
                (self.vertices[i], self.vertices[j])
                for (i, j), var in self.arc_vars.items()
                if solver.value(var)
            ]
        )

        # Eulerian circuit and split at depot
        euler = nx.eulerian_circuit(dg, source=self.depot)
        tours, curr = [], [self.depot]
        for u, v in euler:
            curr.append(v)
            if v == self.depot:
                tours.append(curr)
                curr = [self.depot]
        if len(curr) > 1:
            tours.append(curr)
        return tours
