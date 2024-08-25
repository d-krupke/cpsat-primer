from ._instance import Instance, Solution, Placement
from ortools.sat.python import cp_model
from typing import Optional


class RectanglePackingWithRotationsModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = cp_model.CpModel()

        # Create coordinates for the placement. We need variables for the begin and end of each rectangle.
        # This will also ensure that the rectangles are placed inside the container.
        self.bottom_left_x_vars = [
            self.model.new_int_var(0, instance.container.width, name=f"x1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.bottom_left_y_vars = [
            self.model.new_int_var(0, instance.container.height, name=f"y1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.upper_right_x_vars = [
            self.model.new_int_var(0, instance.container.width, name=f"x2_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.upper_right_y_vars = [
            self.model.new_int_var(0, instance.container.height, name=f"y2_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        # These variables indicate if a rectangle is rotated or not
        self.rotated_vars = [
            self.model.new_bool_var(f"rotated_{i}")
            for i in range(len(instance.rectangles))
        ]
        # Depending on if a rectangle is rotated or not, we have to adjust the width and height variables
        self.width_vars = [
            self.model.new_int_var(0, max(box.width, box.height), name=f"width_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        self.height_vars = [
            self.model.new_int_var(0, max(box.width, box.height), name=f"height_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        # Here we enforce that the width and height variables are correctly set
        for i, box in enumerate(instance.rectangles):
            if box.width > box.height:
                diff = box.width - box.height
                self.model.add(
                    self.width_vars[i] == box.width - self.rotated_vars[i] * diff
                )
                self.model.add(
                    self.height_vars[i] == box.height + self.rotated_vars[i] * diff
                )
            else:
                diff = box.height - box.width
                self.model.add(
                    self.width_vars[i] == box.width + self.rotated_vars[i] * diff
                )
                self.model.add(
                    self.height_vars[i] == box.height - self.rotated_vars[i] * diff
                )

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we don't need them here
        self.x_interval_vars = [
            self.model.new_interval_var(
                start=self.bottom_left_x_vars[i],
                size=self.width_vars[i],
                end=self.upper_right_x_vars[i],
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        self.y_interval_vars = [
            self.model.new_interval_var(
                start=self.bottom_left_y_vars[i],
                size=self.height_vars[i],
                end=self.upper_right_y_vars[i],
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.add_no_overlap_2d(self.x_interval_vars, self.y_interval_vars)

    def _extract_solution(self, solver: cp_model.CpSolver) -> Optional[Solution]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            placements.append(
                Placement(
                    x=solver.value(self.bottom_left_x_vars[i]),
                    y=solver.value(self.bottom_left_y_vars[i]),
                    rotated=bool(solver.value(self.rotated_vars[i])),
                )
            )
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
