from importlib.util import module_from_spec, spec_from_file_location
from itertools import permutations
from pathlib import Path

import pytest


spec = spec_from_file_location(
    "optimization_api_solver",
    Path(__file__).resolve().parents[1] / "examples/optimization_api/app/solver.py",
)
api = module_from_spec(spec)
spec.loader.exec_module(api)


@pytest.mark.parametrize("reverse_edges", [False, True])
def test_node_order_follows_selected_tour(reverse_edges):
    costs = {
        (a, b): 1 if (a, b) in [(0, 2), (2, 1), (1, 0)] else 10
        for a in range(3)
        for b in range(3)
        if a != b
    }
    edges = [
        api.DirectedEdge(source=a, target=b, cost=c) for (a, b), c in costs.items()
    ]
    if reverse_edges:
        edges.reverse()
    instance = api.TspInstance(num_nodes=3, edges=edges)
    solver = api.TspSolver(instance, api.OptimizationParameters(timeout=3))
    solution = solver.solve()
    assert sorted(solution.node_order) == [0, 1, 2]
    tour = solution.node_order
    tour_cost = sum(costs[a, b] for a, b in zip(tour, tour[1:] + tour[:1]))
    oracle = min(
        sum(costs[a, b] for a, b in zip(order, order[1:] + order[:1]))
        for order in permutations(range(3))
    )
    assert tour_cost == oracle == 3
    assert solution.cost == pytest.approx(oracle)
    assert solution.lower_bound == pytest.approx(oracle)
