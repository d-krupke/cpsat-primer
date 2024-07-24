from typing import Callable

from ortools.sat.python import cp_model

from models import TspInstance, OptimizationParameters, TspSolution


class TspSolver:
    def __init__(
        self, tsp_instance: TspInstance, optimization_parameters: OptimizationParameters
    ):
        self.tsp_instance = tsp_instance
        self.optimization_parameters = optimization_parameters
        self.model = cp_model.CpModel()
        self.edge_vars = {
            (edge.source, edge.target): self.model.new_bool_var(
                f"x_{edge.source}_{edge.target}"
            )
            for edge in tsp_instance.edges
        }
        self.model.minimize(
            sum(
                edge.cost * self.edge_vars[(edge.source, edge.target)]
                for edge in tsp_instance.edges
            )
        )
        self.model.add_circuit(
            [(source, target, var) for (source, target), var in self.edge_vars.items()]
        )

    def solve(self, log_callback: Callable[[str], None] | None = None):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.optimization_parameters.timeout
        if log_callback:
            solver.parameters.log_search_progress = True
            solver.log_callback = log_callback
        status = solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return TspSolution(
                node_order=[
                    source
                    for (source, target), var in self.edge_vars.items()
                    if solver.value(var)
                ],
                cost=solver.objective_value,
                lower_bound=solver.best_objective_bound,
            )
        if status == cp_model.INFEASIBLE:
            return TspSolution(
                node_order=None,
                cost=float("inf"),
                lower_bound=float("inf"),
                is_infeasible=True,
            )
        return TspSolution(
            node_order=None,
            cost=float("inf"),
            lower_bound=solver.best_objective_bound,
        )
