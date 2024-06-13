import random
import math
from ortools.sat.python import cp_model

from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional, Callable


class WeightedDirectedGraph(BaseModel):
    num_vertices: int
    edges: Dict[Tuple[int, int], int]


class Tour(BaseModel):
    sequence: list[int]


def generate_random_graph(n: int) -> WeightedDirectedGraph:
    """Generate a random graph with n nodes and n*(n-1) edges."""
    import random

    graph = {}
    for u in range(n):
        for v in range(n):
            if u != v:
                graph[(u, v)] = random.randint(0, 100)
    return WeightedDirectedGraph(num_vertices=n, edges=graph)


def generate_random_geometric_graph(
    n: int, seed=None
) -> Tuple[WeightedDirectedGraph, List[Tuple[int, int]]]:
    random.seed(seed)
    graph = {}
    vertices = list(
        set((random.randint(0, 1000), random.randint(0, 1000)) for _ in range(n))
    )
    n = len(vertices)
    for u in range(n):
        for v in range(n):
            if u != v:
                x1, y1 = vertices[u]
                x2, y2 = vertices[v]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                graph[(u, v)] = int(dist)
    return WeightedDirectedGraph(num_vertices=n, edges=graph), vertices


def resolve_tour_sequence(tour: list[Tuple[int, int]], start: int) -> list[int]:
    successor = {u: v for u, v in tour}
    sequence = [start]
    while True:
        next_node = successor[sequence[-1]]
        if next_node == start:
            break
        sequence.append(next_node)
    return sequence


class _EdgeVars:
    """
    A container for the edge variables in the TSP model.
    """

    def __init__(self, edges: Dict[Tuple[int, int], int], model: cp_model.CpModel):
        self.vars = {(u, v): model.NewBoolVar(f"e_{u}_{v}") for (u, v) in edges.keys()}

    def __getitem__(self, key: Tuple[int, int]):
        return self.vars[key]

    def extract_tour(self, get_value: Callable[[cp_model.LinearExprT], int]) -> Tour:
        tour = [(u, v) for (u, v), var in self.vars.items() if get_value(var)]
        sequence = resolve_tour_sequence(tour, start=0)
        return Tour(sequence=sequence)

    def items(self):
        return self.vars.items()


class TspSolver:
    """
    A simple TSP solver using CP-SAT. It is not necessary efficient and only
    serves as an example.
    """

    def __init__(self, graph: WeightedDirectedGraph):
        self.graph = graph
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self._build_model()

    def _build_model(self):
        self.edge_vars = _EdgeVars(self.graph.edges, self.model)
        circuit = [(u, v, var) for ((u, v), var) in self.edge_vars.items()]
        self.model.AddCircuit(circuit)

        def weight(u, v):
            return self.graph.edges[u, v]

        self.model.Minimize(
            sum(weight(u, v) * var for (u, v), var in self.edge_vars.items())
        )

    def solve(
        self,
        max_time: float = 60.0,
        callback: Optional[cp_model.CpSolverSolutionCallback] = None,
    ) -> Tuple[int, Optional[Tour]]:
        self.solver.parameters.max_time_in_seconds = max_time
        status = self.solver.Solve(self.model, callback)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return status, self.edge_vars.extract_tour(self.solver.Value)
        return status, None
