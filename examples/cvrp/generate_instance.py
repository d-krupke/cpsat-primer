# Generate a random graph with a specified number of nodes.
import itertools
import networkx as nx
import random


def generate_random_euclidean_graph_with_demands(
    n: int,
    min_demand: int = 1,
    max_demand: int = 5,
) -> nx.Graph:
    """
    Generate a random graph with a specified number of nodes.
    The nodes will be integers 0, 1, ..., n-1.
    The graph will be a complete graph with Euclidean distance as edge weight.
    """
    points = [
        (
            random.randint(0, 1_000),
            random.randint(0, 1_000),
            random.randint(min_demand, max_demand),
        )
        for _ in range(n)
    ]
    points[0] = (points[0][0], points[0][1], 0)  # depot with demand 0
    G = nx.Graph()
    # add nodes with demand
    for i, (x, y, demand) in enumerate(points):
        G.add_node(i, pos=(x, y), demand=demand)

    # Complete graph with Euclidean distance as edge weight.
    for i, j in itertools.combinations(range(n), 2):
        p = points[i]
        q = points[j]
        dist = round(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)
        G.add_edge(i, j, weight=dist)
    return G
