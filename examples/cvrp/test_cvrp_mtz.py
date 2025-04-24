import networkx as nx
from ortools.sat.python import cp_model
from .utils import (
    ExpectModelFeasible,
    ExpectModelInfeasible,
    ExpectObjectiveValue,
    assert_optimal,
)

from .cvrp_mtz import CvrpVanillaMtz


def _generate_test_graph(n: int = 5) -> nx.Graph:
    graph = nx.complete_graph(n)
    for i, j in graph.edges:
        graph[i][j]["weight"] = 1
    for i in graph.nodes:
        graph.nodes[i]["demand"] = 1 if i != 0 else 0
    return graph


def test_mtz_feasibility():
    with ExpectModelFeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpVanillaMtz(graph, depot=0, capacity=5, model=model)


def test_mtz_infeasible_zero_capacity():
    with ExpectModelInfeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpVanillaMtz(graph, depot=0, capacity=0, model=model)


def test_mtz_feasible_minimal_capacity():
    with ExpectModelFeasible() as model:
        graph = _generate_test_graph()
        _ = CvrpVanillaMtz(graph, depot=0, capacity=1, model=model)


def test_mtz_optimal_objective():
    with ExpectObjectiveValue(5) as model:
        graph = _generate_test_graph()
        cmc = CvrpVanillaMtz(graph, depot=0, capacity=5, model=model)
        model.minimize(cmc.weight(label="weight"))


def test_mtz_conflicting_arcs():
    with ExpectModelInfeasible() as model:
        graph = _generate_test_graph()
        cmc = CvrpVanillaMtz(graph, depot=0, capacity=5, model=model)
        model.add(cmc.is_arc_used(1, 2) == 1)
        model.add(cmc.is_arc_used(2, 1) == 1)


def test_mtz_conflicting_arcs_direct():
    with ExpectModelFeasible() as model:
        graph = _generate_test_graph()
        cmc = CvrpVanillaMtz(graph, depot=0, capacity=5, model=model)
        model.add(cmc.is_arc_used(0, 2) == 1)
        model.add(cmc.is_arc_used(2, 0) == 1)


def test_mtz_limited_capacity():
    with ExpectObjectiveValue(8) as model:
        graph = _generate_test_graph()
        cvrp = CvrpVanillaMtz(graph, depot=0, capacity=1, model=model)
        model.minimize(cvrp.weight(label="weight"))


def test_mtz_extract_tours():
    graph = _generate_test_graph()
    cvrp = CvrpVanillaMtz(graph, depot=0, capacity=1)
    solver = cp_model.CpSolver()
    assert_optimal(solver=solver, model=cvrp.model)
    tours = cvrp.extract_tours(solver)
    assert len(tours) == 4, f"Expected 4 tours, but got {len(tours)}."
