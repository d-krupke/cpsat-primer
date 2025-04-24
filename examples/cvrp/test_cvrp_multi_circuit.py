"""
Test suite for the MtzBasedFormulation CP-SAT model.

Each test uses a small complete graph of 5 nodes:
  - edge weight = 1,
  - node 0 is the depot (demand = 0),
  - all other nodes demand = 1.
"""

import networkx as nx
from ortools.sat.python import cp_model

from .utils import ExpectFeasible, ExpectInfeasible, ExpectObjective, solve
from .cvrp_mtz import MtzBasedFormulation


def _generate_test_graph(n: int = 5) -> nx.Graph:
    """
    Create a complete graph on n nodes.
    - Every edge gets weight = 1.
    - Node 0 is the depot (demand = 0).
    - All other nodes have demand = 1.
    """
    G = nx.complete_graph(n)
    for i, j in G.edges():
        G[i][j]["weight"] = 1
    for node in G.nodes():
        G.nodes[node]["demand"] = 0 if node == 0 else 1
    return G


def test_mtz_feasible_capacity_5():
    """
    With capacity = 5, one vehicle can visit all customers in one tour → feasible.
    """
    graph = _generate_test_graph()
    with ExpectFeasible() as model:
        MtzBasedFormulation(graph, depot=0, capacity=5, model=model)


def test_mtz_infeasible_capacity_0():
    """
    With capacity = 0, the vehicle cannot serve any customer → infeasible.
    """
    graph = _generate_test_graph()
    with ExpectInfeasible() as model:
        MtzBasedFormulation(graph, depot=0, capacity=0, model=model)


def test_mtz_feasible_capacity_1():
    """
    With capacity = 1, each tour serves exactly one customer → still feasible.
    """
    graph = _generate_test_graph()
    with ExpectFeasible() as model:
        MtzBasedFormulation(graph, depot=0, capacity=1, model=model)


def test_mtz_optimal_weight_capacity_5():
    """
    Minimizing total weight with capacity = 5:
    one loop through all 4 customers → cost = 5.
    """
    graph = _generate_test_graph()
    with ExpectObjective(5) as model:
        mtz = MtzBasedFormulation(graph, depot=0, capacity=5, model=model)
        model.minimize(mtz.weight(label="weight"))


def test_mtz_optimal_weight_capacity_1():
    """
    Minimizing total weight with capacity = 1:
    four separate out‐and‐back tours → cost = 4 × 2 = 8.
    """
    graph = _generate_test_graph()
    with ExpectObjective(8) as model:
        mtz = MtzBasedFormulation(graph, depot=0, capacity=1, model=model)
        model.minimize(mtz.weight(label="weight"))


def test_mtz_conflicting_arcs_infeasible():
    """
    Forbid both directions on a non-depot arc (1↔2):
    violates the MTZ flow constraints → infeasible.
    """
    graph = _generate_test_graph()
    with ExpectInfeasible() as model:
        mtz = MtzBasedFormulation(graph, depot=0, capacity=5, model=model)
        model.add(mtz.is_arc_used(1, 2) == 1)
        model.add(mtz.is_arc_used(2, 1) == 1)


def test_mtz_conflicting_arcs_direct_feasible():
    """
    Force both directions on a depot–customer arc (0↔2):
    allowed by flow + capacity → still feasible.
    """
    graph = _generate_test_graph()
    with ExpectFeasible() as model:
        mtz = MtzBasedFormulation(graph, depot=0, capacity=5, model=model)
        model.add(mtz.is_arc_used(0, 2) == 1)
        model.add(mtz.is_arc_used(2, 0) == 1)


def test_mtz_extract_tours():
    """
    Solve with capacity = 1 (each customer in its own loop),
    then extract_tours should return exactly four [0, i, 0] tours.
    """
    graph = _generate_test_graph()
    model = cp_model.CpModel()
    mtz = MtzBasedFormulation(graph, depot=0, capacity=1, model=model)
    mtz.minimize_weight()
    solver = solve(mtz.model)

    tours = mtz.extract_tours(solver)
    assert len(tours) == 4, f"Expected 4 tours, but got {len(tours)}."
    # Optionally, verify each tour starts and ends at the depot:
    for tour in tours:
        assert tour[0] == 0 and tour[-1] == 0
