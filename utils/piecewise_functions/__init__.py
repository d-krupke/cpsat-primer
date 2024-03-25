"""
This module contains code for piecewise functions, which can be used to approximate
more complex functions for CP-SAT.

The code is under MIT license, and is free to use, modify, or distribute.
Just copy and paste for whatever project you are working on.

https://github.com/d-krupke/cpsat-primer

Author: Dominik Krupke (2024)
"""

from .piecewise_constant_function import (
    PiecewiseConstantFunction,
    PiecewiseConstantConstraint,
    PiecewiseConstantConstraintViaOnlyIf,
)

from .piecewise_linear_function import (
    PiecewiseLinearFunction,
    PiecewiseLinearConstraint,
)

__all__ = [
    "PiecewiseConstantFunction",
    "PiecewiseConstantConstraint",
    "PiecewiseConstantConstraintViaOnlyIf",
    "PiecewiseLinearFunction",
    "PiecewiseLinearConstraint",
]
