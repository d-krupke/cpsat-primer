import networkx as nx
from ortools.sat.python import cp_model


class CvrpVanillaMtz:
    """CVRP via MTZ-based formulation using CP-SAT."""

    def __init__(
        self,
        graph: nx.Graph,
        depot,
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

        # Arc variables for both directions
        self.arc_vars = {
            (i, j): self.model.new_bool_var(f"arc_{i}_{j}")
            for u, v in graph.edges
            for i, j in ((self.index[u], self.index[v]), (self.index[v], self.index[u]))
        }

        # Flow and capacity constraints
        self._add_flow_constraints()
        self._add_capacity_constraints()

    def _add_flow_constraints(self):
        for v in self.vertices:
            if v == self.depot:
                continue
            i = self.index[v]
            nbrs = [self.index[n] for n in self.graph.neighbors(v)]
            incoming = sum(self.arc_vars[(j, i)] for j in nbrs)
            outgoing = sum(self.arc_vars[(i, j)] for j in nbrs)
            self.model.add(incoming == outgoing)
            self.model.add(incoming == 1)

    def _add_capacity_constraints(self):
        self.load = [
            self.model.new_int_var(0, self.capacity, f"load_{i}")
            for i in range(len(self.vertices))
        ]
        for (i, j), var in self.arc_vars.items():
            if j == 0:
                continue
            demand = self.graph.nodes[self.vertices[j]].get(self.demand_label)
            if demand is None or demand <= 0:
                raise ValueError(
                    f"Demand for {self.vertices[j]} must be positive, got {demand}."
                )
            self.model.add(self.load[j] >= self.load[i] + demand).only_enforce_if(var)

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
        dg = nx.DiGraph(
            [
                (self.vertices[i], self.vertices[j])
                for (i, j), var in self.arc_vars.items()
                if solver.value(var)
            ]
        )
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
