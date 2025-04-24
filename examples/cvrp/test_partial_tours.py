import networkx as nx
from .utils import ExpectFeasible, ExpectInfeasible, ExpectObjective

from .partial_tour import PartialTourWithDepot


def _generate_test_graph(n: int = 5) -> nx.Graph:
    graph = nx.complete_graph(n)
    for i, j in graph.edges:
        graph[i][j]["weight"] = 1
    for i in graph.nodes:
        graph.nodes[i]["demand"] = 1 if i != 0 else 0
    return graph


def test_tour():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        _ = PartialTourWithDepot(graph, model)


def test_tour_with_depot():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)


def test_tour_fully_visited():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        for i in graph.nodes:
            model.add(tour.is_visited(i) == 1)


def test_tour_with_capacity():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(5)
        for i in graph.nodes:
            model.add(tour.is_visited(i) == 1)


def test_tour_with_capcity_and_infeasible():
    with ExpectInfeasible() as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(3)
        for i in graph.nodes:
            model.add(tour.is_visited(i) == 1)


def test_tour_with_objective():
    with ExpectObjective(5) as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(4)
        for i in graph.nodes:
            model.add(tour.is_visited(i) == 1)
        model.minimize(tour.tour_length(label="weight"))


def test_partial_tour_with_objective():
    with ExpectObjective(3) as model:
        graph = _generate_test_graph()
        tour = PartialTourWithDepot(graph, model)
        tour.set_depot(0)
        tour.set_capacity(3)
        model.maximize(sum(tour.is_visited(i) for i in graph.nodes if i != 0))
