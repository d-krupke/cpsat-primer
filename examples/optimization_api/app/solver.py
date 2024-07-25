"""
This file contains the implementation of the optimization algorithm
we want to deploy as an API. The actual algorithm is just a simple
implementation of the Traveling Salesman Problem (TSP) using the
`add_circuit` constraint from the CP-SAT solver in OR-Tools.
For more complex optimization algorithms, the algorithm should be
a separate module/project that can be imported here. Having the
algorithm directly in the API code makes the isolated testing
and benchmarking of the algorithm more complicated. Additionally,
if you are working in a team, the API may be maintained by a
different team than the optimization algorithm, so separating
the two can make the development process more efficient.
"""
from typing import Callable

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# A precise definition of the input and output data for the TSP solver.
# ---------------------------------------------------------------------


class DirectedEdge(BaseModel):
    source: int = Field(..., ge=0, description="The source node of the edge.")
    target: int = Field(..., ge=0, description="The target node of the edge.")
    cost: int = Field(..., ge=0, description="The cost of traversing the edge.")


class TspInstance(BaseModel):
    num_nodes: int = Field(
        ..., gt=0, description="The number of nodes in the TSP instance."
    )
    edges: list[DirectedEdge] = Field(
        ..., description="The directed edges of the TSP instance."
    )


class OptimizationParameters(BaseModel):
    timeout: int = Field(
        default=60,
        gt=0,
        description="The maximum time in seconds to run the optimization.",
    )


class TspSolution(BaseModel):
    node_order: list[int] | None = Field(
        ..., description="The order of the nodes in the solution."
    )
    cost: float = Field(..., description="The cost of the solution.")
    lower_bound: float = Field(..., description="The lower bound of the solution.")
    is_infeasible: bool = Field(
        default=False, description="Whether the instance is infeasible."
    )


# ---------------------------------------------------------------------
# The TSP solver implementation using the CP-SAT solver from OR-Tools.
# ---------------------------------------------------------------------


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
