"""
First, we need to generate the instances, if we do not have a ready set of instances.
In this case, we could actually use the TSPLib instances, but let us assume we do not have them.
"""

import networkx as nx
import random
import itertools
import typing
from _utils import GraphInstanceDb
from _conf import INSTANCE_DB


# Generate a random graph with a specified number of nodes.
def generate_random_euclidean_graph(
    n: int,
) -> typing.Tuple[nx.Graph, typing.List[typing.Tuple[int, int]]]:
    """
    Generate a random graph with a specified number of nodes.
    The nodes will be integers 0, 1, ..., n-1.
    The graph will be a complete graph with Euclidean distance as edge weight.
    """
    points = [(random.randint(0, 10_000), random.randint(0, 10_000)) for _ in range(n)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Complete graph with Euclidean distance as edge weight.
    for i, j in itertools.combinations(range(n), 2):
        p = points[i]
        q = points[j]
        dist = round((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
        G.add_edge(i, j, weight=dist)
    return G, points


if __name__ == "__main__":
    assert not INSTANCE_DB.exists(), "Don't accidentally overwrite the instances!"
    graph_db = GraphInstanceDb(INSTANCE_DB)
    for n in [25, 50, 75, 100, 150, 200, 250, 300, 350, 400]:
        for i in range(10):
            G, P = generate_random_euclidean_graph(n)
            graph_db[f"random_euclidean_{n}_{i}"] = G
