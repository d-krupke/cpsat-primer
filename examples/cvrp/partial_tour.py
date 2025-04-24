import networkx as nx
from ortools.sat.python import cp_model


class PartialTourWithDepot:
    """
    A CP-SAT “partial tour” model with an optional depot and capacity.

    Variables:
      - edge_vars[(i, j)]: Bool, 1 if arc i→j is used.
      - node_vars[i]:      Bool, 1 if node i is visited.
      - active_var:        Bool, 1 if this tour is “active” (visits ≥ 1 node).

    Constraints:
      - A single circuit over all nodes + self-loops on unvisited nodes.
      - If inactive (active=0), then no node_vars=1; if active, then ≥1 node_vars=1.
      - Optionally, set a depot: active ⇔ depot visited.
      - Optionally, enforce total demand ≤ capacity.
    """

    def __init__(self, graph: nx.Graph, model: cp_model.CpModel):
        self.graph = graph
        self.model = model

        # Map original node labels → indices 0…n-1
        self.vertex_labels = list(graph.nodes())
        self.vertex_indices = {v: i for i, v in enumerate(self.vertex_labels)}
        n = len(self.vertex_labels)

        # 1) Arc vars + circuit‐arcs list
        self.edge_vars: dict[tuple[int, int], cp_model.BoolVarT] = {}
        arcs: list[tuple[int, int, cp_model.BoolVarT]] = []
        for u, v in graph.edges():
            i, j = self.vertex_indices[u], self.vertex_indices[v]
            var_ij = model.new_bool_var(f"edge_{i}_{j}")
            var_ji = model.new_bool_var(f"edge_{j}_{i}")
            self.edge_vars[(i, j)] = var_ij
            self.edge_vars[(j, i)] = var_ji
            arcs.append((i, j, var_ij))
            arcs.append((j, i, var_ji))

        # 2) Node‐visited vars + self‐loops for unvisited
        self.node_vars: dict[int, cp_model.BoolVarT] = {
            i: model.new_bool_var(f"node_{i}") for i in range(n)
        }
        for i, node_var in self.node_vars.items():
            # if node_var=0 (unvisited), the self-loop must be used
            arcs.append((i, i, node_var.Not()))

        # 3) Enforce exactly one circuit covering all nodes/self-loops
        model.add_circuit(arcs)

        # 4) Active toggle: no visits if inactive; at least one if active
        self.active_var = model.new_bool_var("active")
        for node_var in self.node_vars.values():
            model.add(node_var <= self.active_var)
        model.add(sum(self.node_vars.values()) >= self.active_var)

        # Depot unset until set_depot() is called
        self.depot = None

    def set_depot(self, v):
        """
        Declare v as the depot node.
        Enforces: active_var == node_vars[v].
        """
        idx = self.vertex_indices[v]
        self.depot = v
        self.model.add(self.active_var == self.node_vars[idx])

    def set_capacity(self, capacity: int, label: str = "demand"):
        """
        Enforce sum(demand_i * visited_i) ≤ capacity.
        Assumes graph.nodes[i][label] exists.
        """
        expr = sum(
            self.node_vars[self.vertex_indices[node]] * self.graph.nodes[node][label]
            for node in self.graph.nodes()
        )
        self.model.add(expr <= capacity)

    def is_visited(self, v) -> cp_model.BoolVarT:
        """Return the BoolVar for whether node v is visited."""
        return self.node_vars[self.vertex_indices[v]]

    def is_arc_used(self, u, v) -> cp_model.BoolVarT:
        """Return the BoolVar for whether arc u→v is used."""
        return self.edge_vars[(self.vertex_indices[u], self.vertex_indices[v])]

    def is_active(self) -> cp_model.BoolVarT:
        """Return the BoolVar that toggles tour existence."""
        return self.active_var

    def weight(self, label: str = "weight") -> cp_model.LinearExprT:
        """
        Build a linear expression for the total weight of used arcs.
        Counts both directions explicitly.
        """
        terms = []
        for u, v in self.graph.edges():
            w = self.graph[u][v][label]
            terms.append(self.is_arc_used(u, v) * w)
            terms.append(self.is_arc_used(v, u) * w)
        return sum(terms)

    def extract_tour(self, solver: cp_model.CpSolver, source=None) -> list:
        """
        Reconstruct the active tour as a node list.

        - If un­active, returns [].
        - Otherwise performs DFS from `source` (or depot) and appends depot to close.
        """
        if solver.value(self.active_var) == 0:
            return []

        # Gather used arcs
        used_edges = [
            (self.vertex_labels[i], self.vertex_labels[j])
            for (i, j), var in self.edge_vars.items()
            if solver.value(var) == 1
        ]

        # Build directed subgraph and DFS
        subg = nx.DiGraph()
        subg.add_edges_from(used_edges)
        start = source if source is not None else self.depot
        if start is None:
            tour_nodes = list(nx.dfs_preorder_nodes(subg))
        else:
            tour_nodes = list(nx.dfs_preorder_nodes(subg, source=start))
            tour_nodes.append(start)

        return tour_nodes
