"""
Test suite for PartialTourWithDepot CP-SAT model.

Graph used in all tests: a complete graph on 5 nodes where
  - node 0 is the depot (demand=0),
  - all other nodes have demand=1,
  - every edge has weight=1.
"""

import networkx as nx
from .utils import ExpectModelFeasible, ExpectModelInfeasible, ExpectObjectiveValue
from .partial_tour import PartialTourWithDepot


def _generate_test_graph(n: int = 5) -> nx.Graph:
    """
    Build a complete graph with:
      - uniform edge weight = 1,
      - depot node 0 with demand = 0,
      - all other nodes demand = 1.
    """
    G = nx.complete_graph(n)
    nx.set_edge_attributes(G, 1, "weight")
    for node in G.nodes():
        G.nodes[node]["demand"] = 0 if node == 0 else 1
    return G


def test_initial_partial_tour_feasible():
    """
    A fresh PartialTourWithDepot (no depot or capacity set)
    should be feasible (i.e. the inactive self-loops circuit).
    """
    graph = _generate_test_graph()
    with ExpectModelFeasible() as model:
        PartialTourWithDepot(graph, model)


def test_set_depot_still_feasible():
    """
    Setting a depot makes the tour optional but does not force visits,
    so the model remains feasible.
    """
    graph = _generate_test_graph()
    with ExpectModelFeasible() as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)


def test_force_full_visit_without_capacity():
    """
    Forcing every node (including depot) to be visited,
    without any capacity limit, yields a Hamiltonian circuit → feasible.
    """
    graph = _generate_test_graph()
    with ExpectModelFeasible() as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        for node in graph.nodes():
            model.add(tour.is_visited(node) == 1)


def test_full_visit_with_sufficient_capacity():
    """
    Demand sum = 4; setting capacity = 5 allows full visit → feasible.
    """
    graph = _generate_test_graph()
    with ExpectModelFeasible() as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(5)
        for node in graph.nodes():
            model.add(tour.is_visited(node) == 1)


def test_full_visit_with_insufficient_capacity():
    """
    Demand sum = 4; setting capacity = 3 makes full visit infeasible.
    """
    graph = _generate_test_graph()
    with ExpectModelInfeasible() as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(3)
        for node in graph.nodes():
            model.add(tour.is_visited(node) == 1)


def test_minimize_tour_length():
    """
    With capacity ≥ 4 and forcing all visits,
    the shortest possible circuit has length = 5.
    """
    graph = _generate_test_graph()
    with ExpectObjectiveValue(5) as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(4)
        for node in graph.nodes():
            model.add(tour.is_visited(node) == 1)
        model.minimize(tour.weight(label="weight"))


def test_maximize_customers_visited():
    """
    With capacity = 3, maximize number of customers visited (excluding depot)
    → expect 3.
    """
    graph = _generate_test_graph()
    with ExpectObjectiveValue(3) as model:
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(3)
        model.maximize(
            sum(tour.is_visited(node) for node in graph.nodes() if node != 0)
        )
