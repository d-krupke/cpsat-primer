import itertools
from ortools.sat.python import cp_model
from data_model import DropOff, Instance
import numpy as np


class CapacitatedRouteModel:
    def __init__(self, model: cp_model.CpModel, instance: Instance):
        self.instance = instance
        self.model = model
        self.drop_offs = instance.drop_offs
        self.vehicle_capacity = instance.vehicle_capacity
        self._drop_off_indices = {
            drop_off: i + 1 for i, drop_off in enumerate(self.drop_offs)
        }
        self.cost_matrix = self._create_cost_matrix()
        self._x = self._create_variables()
        self.tour_cost = self._create_cost()

    def _create_cost_matrix(self) -> np.ndarray:
        n = len(self.drop_offs) + 1
        cost_matrix = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            if i == 0:
                c = self.instance.distance(
                    self.instance.depot_location, self.drop_offs[j - 1].location_id
                )
            else:
                c = self.instance.distance(
                    self.drop_offs[i - 1].location_id, self.drop_offs[j - 1].location_id
                )
            c = round(c * 100)
            cost_matrix[i, j] = c
            cost_matrix[j, i] = c
        return cost_matrix

    def _create_variables(self):
        n = len(self.drop_offs) + 1
        x = np.array(
            [[self.model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        )
        # Ensure that we get a feasible solution
        self.model.add_circuit([(i, j, x[i, j]) for i in range(n) for j in range(n)])
        num_nodes_visited = sum(~x[i, i] for i in range(1, n))
        # Enforce that if the depot is left, the capacity is not exceeded.
        # If the depot is not left, no other node is visited.
        self.model.add(num_nodes_visited <= ~x[0, 0] * self.instance.vehicle_capacity)
        return x

    def _create_cost(self):
        n = len(self.drop_offs) + 1
        return sum(
            self._x[i, j] * self.cost_matrix[i, j] for i in range(n) for j in range(n)
        )

    def is_visited(self, drop_off: DropOff):
        i = self._drop_off_indices[drop_off]
        return ~self._x[i, i]

    def is_used(self):
        return ~self._x[0, 0]

    def is_visited_by_index(self, i: int):
        return ~self._x[i + 1, i + 1]

    def get_tour(self, solver: cp_model.CpSolver) -> list[DropOff]:
        # first find out which drop-offs are visited
        visited_drop_offs = [
            i for i in range(1, len(self.drop_offs) + 1) if solver.Value(~self._x[i, i])
        ]
        if not visited_drop_offs:
            return []
        # then find out the edges between the visited drop-offs
        edges = {}
        for i in visited_drop_offs:
            for j in range(1, len(self.drop_offs) + 1):
                if i != j and solver.Value(self._x[i, j]):
                    edges[i] = j
                    break
        # extract the tour sequence
        start_idx = list(set(edges.keys()) - set(edges.values()))[0]
        tour = [self.drop_offs[start_idx - 1]]
        while start_idx in edges:
            start_idx = edges[start_idx]
            tour.append(self.drop_offs[start_idx - 1])
        return tour

    def add_hint(self, drop_offs: list[DropOff], limit_deviation: int | None = None):
        indices = [self._drop_off_indices[drop_off] for drop_off in drop_offs]
        edges = {}
        if indices:
            edges[0] = indices[0]
            edges.update((indices[i], indices[i + 1]) for i in range(len(indices) - 1))
            edges[indices[-1]] = 0
        m = np.zeros(self._x.shape)
        for i, j in edges.items():
            m[i, j] = 1
        # set diagonal to 1
        m += np.eye(self._x.shape[0])
        for i in indices:
            m[i, i] = 0
        if indices:
            m[0, 0] = 0
        # inverse the diagonal
        # m[np.diag_indices_from(m)] = 1 - np.diag(m)
        for i in range(self._x.shape[0]):
            for j in range(self._x.shape[1]):
                self.model.add_hint(self._x[i, j], bool(m[i, j]))
        if limit_deviation:
            # restrict the number of locations that can be covered by another route
            n_to_keep = len(indices) - limit_deviation
            if n_to_keep > 0:
                self.model.add(
                    sum(self.is_visited_by_index(i) for i in indices) >= n_to_keep
                )


class MultiCapacitatedRouteModel:
    def __init__(self, instance: Instance, k: int):
        self.instance = instance
        self.k = k
        self.model = cp_model.CpModel()
        self.routes = [CapacitatedRouteModel(self.model, instance) for _ in range(k)]
        for i, _ in enumerate(instance.drop_offs):
            self.model.add(
                sum(route.is_visited_by_index(i) for route in self.routes) == 1
            )
        self.cost = sum(route.tour_cost for route in self.routes)
        self.model.minimize(self.cost)
        self.solver = cp_model.CpSolver()

    def solve(
        self,
        max_time_in_seconds: float = 15,
        log_search_progess=False,
        relative_gap_limit: float = 0.1,
    ) -> list[list[DropOff]] | None:
        self.solver.parameters.max_time_in_seconds = max_time_in_seconds
        self.solver.parameters.log_search_progress = log_search_progess
        self.solver.parameters.relative_gap_limit = relative_gap_limit
        status = self.solver.solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [route.get_tour(self.solver) for route in self.routes]
        return None

    def add_hint(self, drop_offs: list[list[DropOff]]):
        for route, hints in zip(self.routes, drop_offs, strict=True):
            route.add_hint(hints)
