# dirty hack to import the local module from inside
from generate_instance import generate_random_euclidean_graph_with_demands

# for saving the results easily
from algbench import Benchmark  # pip install algbench


def solve_via_add_circuit(_graph, vehicle_capacity, instance_name):
    from cvrp_circuit import CvrpDuplicatedCircuits

    from ortools.sat.python import cp_model

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
    tours1 = cvrp1.extract_tours(solver)

    return {"objective": solver.objective_value, "bound": solver.best_objective_bound}


def solve_via_multiple_circuits(_graph, vehicle_capacity, instance_name):
    from cvrp_multi_circuit import CvrpMultiCircuit

    from ortools.sat.python import cp_model

    cvrp2 = CvrpMultiCircuit(_graph, depot=0, capacity=vehicle_capacity)
    cvrp2.minimize_weight()
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(cvrp2.model)
    tours2 = cvrp2.extract_tours(solver)

    return {"objective": solver.objective_value, "bound": solver.best_objective_bound}


def solve_via_mtz(_graph, vehicle_capacity, instance_name):
    from cvrp_mtz import CvrpVanillaMtz

    from ortools.sat.python import cp_model

    cvrp3 = CvrpVanillaMtz(_graph, depot=0, capacity=vehicle_capacity)
    cvrp3.minimize_weight()
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(cvrp3.model)
    tours3 = cvrp3.extract_tours(solver)

    return {"objective": solver.objective_value, "bound": solver.best_objective_bound}


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
