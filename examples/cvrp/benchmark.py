import sys
from pathlib import Path

# dirty hack to import this folder as the package `cvrp`, as in example.ipynb
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cvrp.generate_instance import generate_random_euclidean_graph_with_demands

# for saving the results easily
from algbench import Benchmark  # pip install algbench
from ortools.sat.python import cp_model


def _result(solver: cp_model.CpSolver, status) -> dict:
    """Objective and bound only if a solution was found; otherwise CP-SAT
    reports 0.0 for both, which would read as a proven optimum."""
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {
        "status": solver.status_name(status),
        "objective": solver.objective_value if solved else None,
        "bound": solver.best_objective_bound if solved else None,
    }


def solve_via_add_circuit(_graph, vehicle_capacity, instance_name):
    from cvrp.cvrp_circuit import CvrpDuplicatedCircuits

    # solve first model
    cvrp1 = CvrpDuplicatedCircuits(
        _graph, depot=0, vehicle_capacity=vehicle_capacity, num_vehicles=15
    )
    cvrp1.minimize_weight()
    cvrp1.break_symmetries()

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60

    status = solver.solve(cvrp1.model)

    return _result(solver, status)


def solve_via_multiple_circuits(_graph, vehicle_capacity, instance_name):
    from cvrp.cvrp_multi_circuit import CvrpMultiCircuit

    cvrp2 = CvrpMultiCircuit(_graph, depot=0, capacity=vehicle_capacity)
    cvrp2.minimize_weight()
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(cvrp2.model)

    return _result(solver, status)


def solve_via_mtz(_graph, vehicle_capacity, instance_name):
    from cvrp.cvrp_mtz import CvrpVanillaMtz

    cvrp3 = CvrpVanillaMtz(_graph, depot=0, capacity=vehicle_capacity)
    cvrp3.minimize_weight()
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(cvrp3.model)

    return _result(solver, status)


if __name__ == "__main__":
    vehicle_capacity = 15
    benchmark = Benchmark("./.data", hide_output=False)
    for i in range(20):
        var = (i % 3 - 1) * 10
        graph = generate_random_euclidean_graph_with_demands(
            n=50 + var, min_demand=1, max_demand=5
        )
        # Add the three different solving methods to the benchmark
        benchmark.add(
            solve_via_add_circuit,
            instance_name=f"instance_{i}",
            _graph=graph,
            vehicle_capacity=vehicle_capacity,
        )

        benchmark.add(
            solve_via_multiple_circuits,
            instance_name=f"instance_{i}",
            _graph=graph,
            vehicle_capacity=vehicle_capacity,
        )

        benchmark.add(
            solve_via_mtz,
            instance_name=f"instance_{i}",
            _graph=graph,
            vehicle_capacity=vehicle_capacity,
        )

    benchmark.compress()  # Save the results
