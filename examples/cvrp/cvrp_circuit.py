import networkx as nx
from ortools.sat.python import cp_model

from .partial_tour import PartialTourWithDepot


class CvrpModel:
    def __init__(
        self,
        graph: nx.Graph,
        depot,
        vehicle_capacity: int,
        num_vehicles: int,
        model: cp_model.CpModel | None = None,
    ):
        self.graph = graph
        self.vehicle_capacity = vehicle_capacity
        self.model = cp_model.CpModel() if model is None else model
        self.subtours = [
            PartialTourWithDepot(graph, self.model) for _ in range(num_vehicles)
        ]

        for tour in self.subtours:
            tour.set_depot(depot)
            tour.set_capacity(vehicle_capacity)
        # enforce that all nodes are visited
        for v in graph.nodes:
            if v == depot:
                # skip depot
                continue
            self.model.add_exactly_one(tour.is_visited(v) for tour in self.subtours)

        # objective
        self.model.minimize(
            sum(tour.tour_length(label="weight") for tour in self.subtours)
        )

    def extract_tours(self, solver: cp_model.CpSolver) -> list:
        tours = []
        for tour in self.subtours:
            tours.append(tour.extract_tour(solver))
        # remove empty tours
        tours = [tour for tour in tours if len(tour) > 0]
        return tours
