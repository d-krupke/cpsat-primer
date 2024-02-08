"""
This module implements various rectangle packing models with CP-SAT using interval variables and the no-overlap constraint.
It is surprisingly performant.
"""

from ._instance import Instance, Rectangle, Placement, Solution, Container
from .packing_wo_rotations import RectanglePackingWithoutRotationsModel
from .packing_with_rotations import RectanglePackingWithRotationsModel
from .knapsack_with_rotations import RectangleKnapsackWithRotationsModel
from .knapsack_wo_rotations import RectangleKnapsackWithoutRotationsModel
from ._plotting import plot_solution

__all__ = [
    "Instance",
    "Rectangle",
    "Placement",
    "Solution",
    "Container",
    "RectanglePackingWithoutRotationsModel",
    "RectanglePackingWithRotationsModel",
    "RectangleKnapsackWithRotationsModel",
    "RectangleKnapsackWithoutRotationsModel",
    "plot_solution",
]
