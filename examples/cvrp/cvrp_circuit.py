from typing import Hashable
import networkx as nx
from ortools.sat.python import cp_model

from partial_tour import PartialTourWithDepot


class CvrpDuplicatedCircuits:
    """
    Capacitated Vehicle Routing Problem via multiple partial‐tour submodels.

    Each vehicle is modeled as a PartialTourWithDepot, which may be
    “active” (visits ≥1 customer) or inactive.  We enforce:
      1. Exactly one tour visits each customer (depot excluded).
      2. Each tour respects its vehicle_capacity.
      3. Total distance (sum of tour_length) is minimized.

    Attributes:
      model:    The underlying CpModel with all constraints.
      subtours: list of PartialTourWithDepot, one per vehicle.
    """

    def __init__(
        self,
        graph: nx.Graph,
        depot: Hashable,
        vehicle_capacity: int,
        num_vehicles: int,
        weight_label: str = "weight",
        demand_label: str = "demand",
        model: cp_model.CpModel | None = None,
        break_symmetries: bool = True,
    ) -> None:
        self.model = model or cp_model.CpModel()
        # Build one partial tour per vehicle
        self.subtours: list[PartialTourWithDepot] = [
            PartialTourWithDepot(graph, self.model) for _ in range(num_vehicles)
        ]

        # Configure each tour: set depot and capacity
        for tour in self.subtours:
            tour.set_depot(depot)
            tour.set_capacity(vehicle_capacity, label=demand_label)

        # Enforce that each customer (non-depot) is visited exactly once
        for node in graph.nodes():
            if node == depot:
                continue
            self.model.add_exactly_one(
                *(tour.is_visited(node) for tour in self.subtours)
            )
        if break_symmetries:
            self.break_symmetries()

    def break_symmetries(self, also_break_over_weight: bool = False) -> None:
        """
        Break symmetries. This does not always help, but it can.
        """
        for tour1, tour2 in zip(self.subtours, self.subtours[1:]):
            # tour1 should have higher priority than tour2
            self.model.add(tour1.is_active() >= tour2.is_active())
            # break on weight too (not necessarily always better)
            if also_break_over_weight:
                self.model.add(tour1.weight() >= tour2.weight())

    def weight(self, label: str = "weight") -> cp_model.LinearExprT:
        """
        Return the total weight of all tours.

        Args:
          label:  The label for the weight attribute in the graph.

        Returns:
          A linear expression representing the total weight.
        """
        return sum(tour.weight(label=label) for tour in self.subtours)

    def minimize_weight(self, label: str = "weight") -> None:
        """
        Set the objective to minimize the total weight of all tours.

        Args:
          label:  The label for the weight attribute in the graph.
        """
        self.model.minimize(self.weight(label=label))

    def extract_tours(
        self,
        solver: cp_model.CpSolver,
        source: Hashable | None = None,
    ) -> list[list[Hashable]]:
        """
        After solving, collect the node sequence for each active tour.

        Args:
          solver:  A CpSolver that has solved self.model.
          source:  Starting node for DFS (defaults to depot).

        Returns:
          A list of tours (each a list of node labels), one per active vehicle.
        """
        tours: list[list[Hashable]] = []
        for tour in self.subtours:
            if solver.value(tour.is_active()):
                tours.append(tour.extract_tour(solver, source=source))
        return tours
