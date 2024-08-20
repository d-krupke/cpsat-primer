from ._instance import Instance, Solution, Placement
from ortools.sat.python import cp_model
from typing import Optional, Tuple


class RectangleSqueezingWithoutRotationsModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = cp_model.CpModel()

        max_width = instance.container.width
        max_height = instance.container.height

        self.box_width_var = self.model.NewIntVar(0, max_width, "box_width")
        self.box_height_var = self.model.NewIntVar(0, max_height, "box_height")

        # We have to create the variable for the bottom left corner of the boxes.
        # We directly limit their range, such that the boxes are inside the container
        self.x_vars = [
            self.model.NewIntVar(0, max_width - box.width, name=f"x1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        for i, x in enumerate(self.x_vars):
            self.model.Add(x + self.instance.rectangles[i].width <= self.box_width_var)
        self.y_vars = [
            self.model.NewIntVar(0, max_height - box.height, name=f"y1_{i}")
            for i, box in enumerate(instance.rectangles)
        ]
        for i, y in enumerate(self.y_vars):
            self.model.Add(
                y + self.instance.rectangles[i].height <= self.box_height_var
            )

        # Interval variables are actually more like constraint containers, that are then passed to the no overlap constraint
        # Note that we could also make size and end variables, but we don't need them here
        x_interval_vars = [
            self.model.NewFixedSizeIntervalVar(
                start=self.x_vars[i],  # the x value of the bottom left corner
                size=box.width,  # the width of the rectangle
                name=f"x_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        y_interval_vars = [
            self.model.NewFixedSizeIntervalVar(
                start=self.y_vars[i],  # the y value of the bottom left corner
                size=box.height,  # the height of the rectangle
                name=f"y_interval_{i}",
            )
            for i, box in enumerate(instance.rectangles)
        ]
        # Enforce that no two rectangles overlap
        self.model.AddNoOverlap2D(x_interval_vars, y_interval_vars)
        self.model.Minimize(self.box_width_var + self.box_height_var)

    def _extract_solution(
        self, solver: cp_model.CpSolver
    ) -> Optional[Tuple[Solution, Tuple[int, int]]]:
        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        placements = []
        for i, box in enumerate(self.instance.rectangles):
            x = solver.Value(self.x_vars[i])
            y = solver.Value(self.y_vars[i])
            placements.append(Placement(x=x, y=y))
        return Solution(placements=placements), (
            solver.Value(self.box_width_var),
            solver.Value(self.box_height_var),
        )

    def solve(self, time_limit: float = 900.0):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = time_limit
        self.status = solver.Solve(self.model)
        self.solution = self._extract_solution(solver)
        self.upper_bound = solver.BestObjectiveBound()
        self.objective_value = solver.ObjectiveValue()
        return self.status

    def is_infeasible(self):
        return self.status == cp_model.INFEASIBLE

    def is_feasible(self):
        return self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
