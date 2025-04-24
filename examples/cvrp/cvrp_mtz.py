import networkx as nx
from ortools.sat.python import cp_model


class MtzBasedFormulation:
    def __init__(
        self,
        graph: nx.Graph,
        depot,
        vehicle_capacity: int,
        demand_label: str = "demand",
        model: cp_model.CpModel | None = None,
    ):
        self.graph = graph
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.model = cp_model.CpModel() if model is None else model
        self.demand_label = demand_label

        self.vertex_labels = list(graph.nodes())
        # move depot to the first position
        if self.vertex_labels[0] != depot:
            self.vertex_labels.remove(depot)
            self.vertex_labels.insert(0, depot)
        self.vertex_indices = {i: v for i, v in enumerate(self.vertex_labels)}

        # Arc Variables
        self.arc_vars = {}
        for v, w in graph.edges:
            i = self.vertex_indices[v]
            j = self.vertex_indices[w]
            assert i != j, "The shouldn't be any self-loops in the graph."
            x1 = self.model.new_bool_var(f"arc_{i}_{j}")
            x2 = self.model.new_bool_var(f"arc_{j}_{i}")
            self.arc_vars[(i, j)] = x1
            self.arc_vars[(j, i)] = x2

        self._enforce_flow()
        self._enforce_capacity_via_only_if()

    def _enforce_flow(self):
        for v in self.graph.nodes:
            if v == self.depot:
                continue
            # flow conservation
            nbrs = [self.vertex_indices[nbr] for nbr in self.graph.neighbors(v)]
            i = self.vertex_indices[v]
            assert i not in nbrs
            incoming = sum(self.arc_vars[(j, i)] for j in nbrs)
            outgoing = sum(self.arc_vars[(i, j)] for j in nbrs)
            self.model.add(incoming == outgoing)
            self.model.add(incoming == 1)

    def _enforce_capacity_via_only_if(self):
        # Capacity Variables
        self.capacity_vars = [
            self.model.new_int_var(0, self.vehicle_capacity, f"capacity_{i}")
            for i in range(len(self.vertex_labels))
        ]
        # Capacity Constraints
        for (i, j), x in self.arc_vars.items():
            if j == 0:
                continue  # depot
            # We only need to propagate the used capacity from the previous node, the variable limit will enforce the capacity
            d = self.graph.nodes[self.vertex_labels[j]][self.demand_label]
            if d <= 0:
                raise ValueError(
                    f"Demand for node {self.vertex_labels[j]} is {d}, but it must be positive for MTZ-based formulation."
                )

            self.model.add(
                self.capacity_vars[j] >= self.capacity_vars[i] + d
            ).only_enforce_if(x)

    def is_arc_used(self, v, w) -> cp_model.BoolVarT:
        return self.arc_vars[(self.vertex_indices[v], self.vertex_indices[w])]

    def weight(self, label: str = "weight") -> cp_model.LinearExprT:
        # we get the length from the edge labels on the graph
        length = sum(
            self.arc_vars[(i, j)]
            * self.graph[self.vertex_labels[i]][self.vertex_labels[j]][label]
            + self.arc_vars[(j, i)]
            * self.graph[self.vertex_labels[i]][self.vertex_labels[j]][label]
            for i, j in self.graph.edges
        )
        return length

    def extract_tours(self, solver: cp_model.CpSolver) -> list:
        # computer the euler tour and then split it at the depot to get the subtours.
        graph = nx.DiGraph()
        for (i, j), x in self.arc_vars.items():
            if solver.value(x) == 1:
                graph.add_edge(self.vertex_labels[i], self.vertex_labels[j])
        # get the euler tour
        tour = list(nx.eulerian_circuit(graph, source=self.depot))
        # split the tour at the depot
        tours = []
        current_tour = []
        for i, j in tour:
            if i != self.depot or len(current_tour) == 0:
                current_tour.append(i)
            if j == self.depot:
                if len(current_tour) > 0:
                    current_tour.append(j)
                    tours.append(current_tour)
                    current_tour = []
        if len(current_tour) > 0:  # Handle case where last tour doesn't end at depot
            tours.append(current_tour)
        return tours
