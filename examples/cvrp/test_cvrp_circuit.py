import networkx as nx
from .cvrp_circuit import CvrpModel
from .utils import ExpectFeasible, ExpectInfeasible, ExpectObjective


def _generate_test_graph(n: int = 5) -> nx.Graph:
    graph = nx.complete_graph(n)
    for i, j in graph.edges:
        graph[i][j]["weight"] = 1
    for i in graph.nodes:
        graph.nodes[i]["demand"] = 1 if i != 0 else 0
    return graph


def test_cvrp_feasible_with_sufficient_capacity():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpModel(graph, depot=0, vehicle_capacity=5, num_vehicles=2, model=model)


def test_cvrp_infeasible_with_insufficient_capacity():
    with ExpectInfeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpModel(graph, depot=0, vehicle_capacity=1, num_vehicles=2, model=model)


def test_cvrp_feasible_with_enough_vehicles():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpModel(graph, depot=0, vehicle_capacity=1, num_vehicles=4, model=model)


def test_cvrp_objective_value():
    with ExpectObjective(5) as model:
        graph = _generate_test_graph()
        _ = CvrpModel(graph, depot=0, vehicle_capacity=5, num_vehicles=2, model=model)


def test_cvrp_infeasible_with_multi_visit():
    with ExpectInfeasible() as model:
        graph = _generate_test_graph()
        cvrp = CvrpModel(
            graph, depot=0, vehicle_capacity=5, num_vehicles=2, model=model
        )
        model.add(cvrp.subtours[0].is_visited(1) == 1)
        model.add(cvrp.subtours[1].is_visited(1) == 1)


def test_cvrp_objective_with_many_vehicles():
    with ExpectObjective(8) as model:
        graph = _generate_test_graph()
        _ = CvrpModel(graph, depot=0, vehicle_capacity=1, num_vehicles=7, model=model)
