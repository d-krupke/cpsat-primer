"""
This module implements a simple TSP solver using CP-SAT.
It serves as a basic example and is not optimized for performance.
We ensure that we can access the solver and pass a callback to it,
which is necessary for embedding it in the solver process (for extracting data during the search).
"""

import random
import math
from ortools.sat.python import cp_model
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional, Callable


class WeightedDirectedGraph(BaseModel):
    """
    Represents a weighted directed graph.

    Attributes:
        num_vertices (int): The number of vertices in the graph.
        edges (Dict[Tuple[int, int], int]): A dictionary of edges with weights.
    """

    num_vertices: int
    edges: Dict[Tuple[int, int], int]


class Tour(BaseModel):
    """
    Represents a tour in the TSP problem.

    Attributes:
        sequence (List[int]): The sequence of vertices in the tour.
    """

    sequence: List[int]


def generate_random_graph(num_vertices: int) -> WeightedDirectedGraph:
    """
    Generate a random graph with specified number of vertices and random weights.

    Args:
        num_vertices (int): The number of vertices in the graph.

    Returns:
        WeightedDirectedGraph: The generated random graph.
    """
    edges = {
        (u, v): random.randint(0, 100)
        for u in range(num_vertices)
        for v in range(num_vertices)
        if u != v
    }
    return WeightedDirectedGraph(num_vertices=num_vertices, edges=edges)


def generate_random_geometric_graph(
    num_vertices: int, seed: Optional[int] = None
) -> Tuple[WeightedDirectedGraph, List[Tuple[int, int]]]:
    """
    Generate a random geometric graph with vertices placed in a 2D plane.

    Args:
        num_vertices (int): The number of vertices in the graph.
        seed (Optional[int]): The seed for random number generator.

    Returns:
        Tuple[WeightedDirectedGraph, List[Tuple[int, int]]]: The generated graph and the list of vertices.
    """
    random.seed(seed)
    vertices = list(
        set(
            (random.randint(0, 1000), random.randint(0, 1000))
            for _ in range(num_vertices)
        )
    )
    num_vertices = len(vertices)
    edges = {}
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u != v:
                x1, y1 = vertices[u]
                x2, y2 = vertices[v]
                distance = int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                edges[(u, v)] = distance
    return WeightedDirectedGraph(num_vertices=num_vertices, edges=edges), vertices


def resolve_tour_sequence(
    tour_edges: List[Tuple[int, int]], start_vertex: int
) -> List[int]:
    """
    Resolve the sequence of vertices in the tour from a list of tour edges.

    Args:
        tour_edges (List[Tuple[int, int]]): The list of edges in the tour.
        start_vertex (int): The starting vertex of the tour.

    Returns:
        List[int]: The sequence of vertices in the tour.
    """
    successor = {u: v for u, v in tour_edges}
    sequence = [start_vertex]
    while True:
        next_vertex = successor[sequence[-1]]
        if next_vertex == start_vertex:
            break
        sequence.append(next_vertex)
    return sequence


class EdgeVariables:
    """
    A container for the edge variables in the TSP model.
    """

    def __init__(self, edges: Dict[Tuple[int, int], int], model: cp_model.CpModel):
        """
        Initialize EdgeVariables with the given edges and model.

        Args:
            edges (Dict[Tuple[int, int], int]): The edges with weights.
            model (cp_model.CpModel): The CP-SAT model.
        """
        self.vars = {
            (u, v): model.new_bool_var(f"edge_{u}_{v}") for (u, v) in edges.keys()
        }

    def __getitem__(self, key: Tuple[int, int]):
        return self.vars[key]

    def extract_tour(self, get_value: Callable[[cp_model.LinearExpr], int]) -> Tour:
        """
        Extract the tour from the solved model.

        Args:
            get_value (Callable[[cp_model.LinearExpr], int]): Function to get the value of a variable.

        Returns:
            Tour: The tour with the sequence of vertices.
        """
        tour_edges = [(u, v) for (u, v), var in self.vars.items() if get_value(var)]
        sequence = resolve_tour_sequence(tour_edges, start_vertex=0)
        return Tour(sequence=sequence)

    def items(self):
        return self.vars.items()


class TspSolver:
    """
    A simple TSP solver using CP-SAT. It serves as an example and is not necessarily efficient.
    """

    def __init__(self, graph: WeightedDirectedGraph):
        """
        Initialize the TSP solver with the given graph.

        Args:
            graph (WeightedDirectedGraph): The graph to solve the TSP for.
        """
        self.graph = graph
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self._build_model()

    def _build_model(self):
        """
        Build the CP-SAT model for the TSP problem.
        """
        self.edge_vars = EdgeVariables(self.graph.edges, self.model)
        circuit = [(u, v, var) for ((u, v), var) in self.edge_vars.items()]
        self.model.add_circuit(circuit)

        def weight(u, v):
            return self.graph.edges[u, v]

        self.model.minimize(
            sum(weight(u, v) * var for (u, v), var in self.edge_vars.items())
        )

    def solve(
        self,
        max_time: float = 60.0,
        callback: Optional[cp_model.CpSolverSolutionCallback] = None,
    ) -> Tuple[int, Optional[Tour]]:
        """
        Solve the TSP problem.

        Args:
            max_time (float): The maximum time to solve the problem.
            callback (Optional[cp_model.CpSolverSolutionCallback]): A callback for the solver.

        Returns:
            Tuple[int, Optional[Tour]]: The status of the solver and the tour if found.
        """
        self.solver.parameters.max_time_in_seconds = max_time
        status = self.solver.solve(self.model, callback)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return status, self.edge_vars.extract_tour(self.solver.Value)
        return status, None
