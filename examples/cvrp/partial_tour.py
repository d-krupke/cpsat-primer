import networkx as nx
from ortools.sat.python import cp_model


class PartialTourWithDepot:
    def __init__(self, graph: nx.Graph, model: cp_model.CpModel):
        self.graph = graph
        self.model = model
        self.vertex_labels = list(graph.nodes())
        self.vertex_indices = {i: v for i, v in enumerate(self.vertex_labels)}
        # x[i,j] is true if the edge (i,j) is used in the tour
        self.edge_vars = {}
        self.depot = None
        arcs = []
        for v, w in graph.edges():
            i = self.vertex_indices[v]
            j = self.vertex_indices[w]
            x1 = model.new_bool_var(f"edge_{i}_{j}")
            x2 = model.new_bool_var(f"edge_{j}_{i}")
            self.edge_vars[(i, j)] = x1
            self.edge_vars[(j, i)] = x2
            arcs.append((i, j, x1))
            arcs.append((j, i, x2))
        # x[v] is true if node v is visited
        self.node_vars = {
            i: model.new_bool_var(f"node_{i}") for i in range(len(self.vertex_labels))
        }
        # we have to use ~x[v] because a True value on the self-loop means that the node is not visited
        arcs.extend([(i, i, ~x) for i, x in self.node_vars.items()])
        # enforce a circuit
        model.add_circuit(arcs)

        self.active_var = model.new_bool_var("active")
        # if not active, then all nodes must be unvisited
        for i in range(len(self.vertex_labels)):
            model.add(self.node_vars[i] <= self.active_var)
        # if active is true, then some node must be visited
        model.add(sum(self.node_vars.values()) >= self.active_var)

    def set_depot(self, v):
        self.depot = v
        self.model.add(self.is_active() == self.is_visited(v))

    def set_capacity(self, capacity: int, label: str = "demand"):
        # we get the capacity from the node labels on the graph
        used_capacity = sum(
            self.is_visited(v) * self.graph.nodes[v][label] for v in self.graph.nodes
        )
        self.model.add(used_capacity <= capacity)

    def is_visited(self, v) -> cp_model.BoolVarT:
        return self.node_vars[self.vertex_indices[v]]

    def is_arc_used(self, v, w) -> cp_model.BoolVarT:
        return self.edge_vars[(self.vertex_indices[v], self.vertex_indices[w])]

    def is_active(self) -> cp_model.BoolVarT:
        return self.active_var

    def tour_length(self, label: str = "weight") -> cp_model.LinearExprT:
        # we get the length from the edge labels on the graph
        length = sum(
            self.is_arc_used(v, w) * self.graph[v][w][label]
            + self.is_arc_used(w, v) * self.graph[v][w][label]
            for v, w in self.graph.edges
        )
        return length

    def extract_tour(self, solver: cp_model.CpSolver, source=None) -> list:
        if solver.value(self.is_active()) == 0:
            return []
        used_arcs = []
        for (i, j), x in self.edge_vars.items():
            if solver.value(x) == 1:
                used_arcs.append((self.vertex_labels[i], self.vertex_labels[j]))
        # extract tour via networkx
        subgraph = nx.DiGraph()
        subgraph.add_edges_from(used_arcs)
        if source is None:
            source = self.depot
        if source is not None:
            tour = list(nx.dfs_preorder_nodes(subgraph, source=source))
        else:
            tour = list(nx.dfs_preorder_nodes(subgraph))
        tour.append(self.depot)  # close the tour
        return tour
