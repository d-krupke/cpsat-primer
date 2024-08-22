from ._instance import Instance, Solution, Placement
from ortools.sat.python import cp_model
from typing import Optional


class RectangleKnapsackWithoutRotationsModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = cp_model.CpModel()

        # We have to create the variable for the bottom left corner of the boxes.
        # We directly limit their range, such that the boxes are inside the container
        self.x_vars = [
            self.model.new_int_var(
                0, instance.container.width - box.width, name=f"x1_{i}"
            )
            for i, box in enumerate(instance.rectangles)
        ]
        self.y_vars = [
            self.model.new_int_var(
                0, instance.container.height - box.height, name=f"y1_{i}"
            )
            for i, box in enumerate(instance.rectangles)
        ]

        # The packed variable is a boolean variable that is true if the rectangle is packed
        self.packed_vars = [
            self.model.new_bool_var(f"packed_{i}")
            for i in range(len(instance.rectangles))
        ]

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we don't need them here
        x_interval_vars = [
            self.model.new_optional_fixed_size_interval_var(
                start=self.x_vars[i],  # the x value of the bottom left corner
                size=box.width,  # the width of the rectangle
                is_present=self.packed_vars[i],  # whether the rectangle is packed
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        y_interval_vars = [
            self.model.new_optional_fixed_size_interval_var(
                start=self.y_vars[i],  # the y value of the bottom left corner
                size=box.height,  # the height of the rectangle
                is_present=self.packed_vars[i],  # whether the rectangle is packed
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.add_no_overlap_2d(x_interval_vars, y_interval_vars)

        # maximize the number of packed rectangles
        self.model.maximize(
            sum(
                box.value * self.packed_vars[i]
                for i, box in enumerate(instance.rectangles)
            )
        )

    def _extract_solution(self, solver: cp_model.CpSolver) -> Optional[Solution]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            if solver.value(self.packed_vars[i]):
                x = solver.value(self.x_vars[i])
                y = solver.value(self.y_vars[i])
                placements.append(Placement(x=x, y=y))
            else:
                placements.append(None)
        return Solution(placements=placements)

    def solve(self, time_limit: float = 900.0, opt_tol: float = 0.01):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.relative_gap_limit = opt_tol
        self.status = solver.solve(self.model)
        self.solution = self._extract_solution(solver)
        self.upper_bound = solver.best_objective_bound
        self.objective_value = solver.objective_value
        return self.status
