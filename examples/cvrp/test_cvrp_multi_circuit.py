import networkx as nx
from .utils import ExpectFeasible, ExpectInfeasible, ExpectObjective

from .cvrp_multi_circuit import CapacitatedMultiCircuit


def _generate_test_graph(n: int = 5) -> nx.Graph:
    graph = nx.complete_graph(n)
    for i, j in graph.edges:
        graph[i][j]["weight"] = 1
    for i in graph.nodes:
        graph.nodes[i]["demand"] = 1 if i != 0 else 0
    return graph


def test_multicircuit_feasibility():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        _ = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=5, model=model)


def test_multicircuit_infeasible_zero_capacity():
    with ExpectInfeasible() as model:
        graph = _generate_test_graph()
        _ = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=0, model=model)


def test_multicircuit_feasible_minimal_capacity():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        _ = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=1, model=model)


def test_multicircuit_optimal_objective():
    with ExpectObjective(5) as model:
        graph = _generate_test_graph()
        cmc = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=5, model=model)
        model.minimize(cmc.weight(label="weight"))


def test_multicircuit_conflicting_arcs():
    with ExpectInfeasible() as model:
        graph = _generate_test_graph()
        cmc = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=5, model=model)
        model.add(cmc.is_arc_used(1, 2) == 1)
        model.add(cmc.is_arc_used(2, 1) == 1)


def test_multicircuit_conflicting_arcs_direct():
    with ExpectFeasible() as model:
        graph = _generate_test_graph()
        cmc = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=5, model=model)
        model.add(cmc.is_arc_used(0, 2) == 1)
        model.add(cmc.is_arc_used(2, 0) == 1)


def test_multicircuit_limited_capacity():
    with ExpectObjective(8) as model:
        # test multi visit
        graph = _generate_test_graph()
        cvrp = CapacitatedMultiCircuit(graph, depot=0, vehicle_capacity=1, model=model)
        model.minimize(cvrp.weight(label="weight"))
