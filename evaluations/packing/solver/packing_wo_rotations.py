from ._instance import Instance, Solution, Placement
from ortools.sat.python import cp_model
from typing import Optional


class RectanglePackingWithoutRotationsModel:
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

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we don't need them here
        x_interval_vars = [
            self.model.new_fixed_size_interval_var(
                start=self.x_vars[i],  # the x value of the bottom left corner
                size=box.width,  # the width of the rectangle
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        y_interval_vars = [
            self.model.new_fixed_size_interval_var(
                start=self.y_vars[i],  # the y value of the bottom left corner
                size=box.height,  # the height of the rectangle
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.add_no_overlap_2d(x_interval_vars, y_interval_vars)

    def _extract_solution(self, solver: cp_model.CpSolver) -> Optional[Solution]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            x = solver.value(self.x_vars[i])
            y = solver.value(self.y_vars[i])
            placements.append(Placement(x=x, y=y))
        return Solution(placements=placements)

    def solve(self, time_limit: float = 900.0):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = time_limit
        self.status = solver.solve(self.model)
        self.solution = self._extract_solution(solver)
        return self.status

    def is_infeasible(self):
        return self.status == cp_model.INFEASIBLE

    def is_feasible(self):
        return self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
