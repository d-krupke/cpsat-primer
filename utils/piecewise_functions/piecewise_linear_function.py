"""
This file implements piecewise linear functions and constraints for CP-SAT.
Because CP-SAT only supports integer values, the functions can only be used as lower or upper bounds,
but not for equality constraints as they would most likely not be satisfiable.

The code is under MIT license, and is free to use, modify, or distribute.
Just copy and paste for whatever project you are working on.

https://github.com/d-krupke/cpsat-primer

Author: Dominik Krupke (2024)
"""

from typing import List, Optional, Tuple
from ortools.sat.python import cp_model
import bisect
import math
import typing
from pydantic import BaseModel, model_validator
from scipy.spatial import ConvexHull


class PiecewiseLinearFunction(BaseModel):
    """
    A piecewise linear function defined by a list of x and y values.
    The function is defined by the line segments connecting the points (x[i], y[i]) and (x[i+1], y[i+1]).
    This class defines the function and provides methods to evaluate it and check properties.
    As a Pydantic model, it also provides validation and serialization.

    This is a version specifically for CP-SAT, allowing only integer values.

    https://en.wikipedia.org/wiki/Piecewise_linear_function
    https://en.wikipedia.org/wiki/Linear_interpolation
    """

    xs: List[int]
    ys: List[int]

    @model_validator(mode="after")
    def validate(self):
        if len(self.xs) != len(self.ys):
            raise ValueError(
                "The number of x values must be equal to the number of y values"
            )
        if any(x1 >= x2 for x1, x2 in zip(self.xs, self.xs[1:])):
            raise ValueError("The x values must be strictly increasing")
        return self

    def is_defined_for(self, x: int):
        return self.xs[0] <= x <= self.xs[-1]

    def get_bounds(self) -> Tuple[int, int]:
        return (self.xs[0], self.xs[-1])

    def __call__(self, x: int) -> float:
        if not self.is_defined_for(x):
            raise ValueError(f"The function is not defined for x={x}")
        if x == self.xs[-1]:
            return self.ys[-1]
        i = bisect.bisect_right(self.xs, x) - 1
        return self.ys[i] + (self.ys[i + 1] - self.ys[i]) * (x - self.xs[i]) / (
            self.xs[i + 1] - self.xs[i]
        )

    def get_segment_gradients(self):
        """
        Returns the gradients of the segments of the piecewise linear function.
        """
        return [
            (y2 - y1) / (x2 - x1)
            for x1, x2, y1, y2 in zip(self.xs, self.xs[1:], self.ys, self.ys[1:])
        ]

    def is_convex(self, upper_bound: bool = True):
        """
        Returns true if the area under the function is convex.
        Allows optimization if the function is an upper bound.

        If `upper_bound` is true, the function is assumed to be an upper bound.
        Then the area is below the function and the function is convex if the gradients are not increasing.

        If `upper_bound` is false, the function is assumed to be a lower bound.
        Then the area is above the function and the function is convex if the gradients are not decreasing.
        """
        gradients = self.get_segment_gradients()
        if upper_bound:
            # Gradients should not be increasing for a convex upper bound
            return all(g1 >= g2 for g1, g2 in zip(gradients, gradients[1:]))
        # Gradients should not be decreasing for a convex lower bound
        return all(g1 <= g2 for g1, g2 in zip(gradients, gradients[1:]))

    def num_segments(self) -> int:
        """
        Returns the number of segments in the piecewise linear function.
        """
        return len(self.xs) - 1

    def segments(
        self,
    ) -> typing.Iterable[typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]]:
        """
        Returns the segments of the piecewise linear function.
        """
        return (
            ((self.xs[i], self.ys[i]), (self.xs[i + 1], self.ys[i + 1]))
            for i in range(self.num_segments())
        )


def test_piecewise_linear_function():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    assert round(f(0)) == 0
    assert round(f(5)) == 5
    assert round(f(10)) == 10
    assert round(f(16)) == 7
    assert round(f(20)) == 5
    assert f.is_convex()


def are_colinear(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    """
    Check if three points are colinear.
    """
    assert p0[0] < p1[0] < p2[0], "Points are sorted"
    # Check if the slopes are equal. We do not want any rounding errors, so we use
    # the cross product instead of the division.
    return (p1[1] - p0[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p1[0] - p0[0])


def test_are_colinear():
    assert are_colinear((0, 0), (10, 10), (20, 20))
    assert are_colinear((0, 1), (10, 11), (20, 21))
    assert not are_colinear((0, 0), (10, 10), (20, 21))
    assert not are_colinear((0, 0), (10, 10), (20, 19))


def minimize_piecewise_linear_function(
    f: PiecewiseLinearFunction,
) -> PiecewiseLinearFunction:
    """
    Removes redundant segments from a piecewise linear function.
    """
    redundant_indices = [
        i
        for i in range(1, len(f.xs) - 1)
        if are_colinear(
            (f.xs[i - 1], f.ys[i - 1]),
            (f.xs[i], f.ys[i]),
            (f.xs[i + 1], f.ys[i + 1]),
        )
    ]
    if not redundant_indices:
        return f.model_copy(deep=True)
    xs = [x for i, x in enumerate(f.xs) if i not in redundant_indices]
    ys = [y for i, y in enumerate(f.ys) if i not in redundant_indices]
    return PiecewiseLinearFunction(xs=xs, ys=ys)


def get_convex_envelope(
    f: PiecewiseLinearFunction, upper_bound: bool = True
) -> PiecewiseLinearFunction:
    """
    The convex envelope of a function is the tightest convex function that is
    an upper resp. lower bound of the function. For a piecewise linear function,
    this can be computed by removing all segments that are not convex or simply
    computing the convex hull of the points.

    As convex functions are much easier to handle in optimization problems, this
    function can be added as redundant constraint to help the solver bound the
    possible values of the function.

    Args:
        f: The piecewise linear function
        upper_bound: If true, the convex envelope is an upper bound, otherwise a lower bound.

    Returns:
        The convex envelope of the function.
    """
    f = minimize_piecewise_linear_function(f)
    if f.is_convex(upper_bound=upper_bound):
        return f.model_copy(deep=True)
    # Add two points at the bottom left and bottom right to ensure the convex hull
    # does not contain points below the function.
    ys = list(f.ys)
    if not upper_bound:
        # To get the convex envelope for a lower bound, we need to flip the function
        ys = [-y for y in ys]
    lower_left = (f.xs[0], min(f.ys) - 1)
    lower_right = (f.xs[-1], min(f.ys) - 1)
    ch = ConvexHull(list(zip(f.xs, f.ys)) + [lower_left, lower_right])
    xs = [x for i, x in enumerate(f.xs) if i in ch.vertices]
    ys = [y if upper_bound else -y for i, y in enumerate(f.ys) if i in ch.vertices]
    f_ = PiecewiseLinearFunction(xs=xs, ys=ys)
    assert f_.is_convex(upper_bound=upper_bound)
    return f_


def test_get_upper_bounding_convex_envelope():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    g = get_convex_envelope(f, upper_bound=True)
    assert g.is_convex()
    assert all(g(x) >= f(x) for x in range(21))
    assert g(0) == 0
    assert g(10) == 25
    assert g(20) == 50
    g_ = get_convex_envelope(f, upper_bound=False)
    assert g_.is_convex(upper_bound=False)
    assert all(g_(x) <= f(x) for x in range(21))
    assert g_(0) == 0
    assert g_(10) == 10
    assert g_(20) == 50


def split_into_convex_segments(
    f: PiecewiseLinearFunction, upper_bound: bool = True
) -> List[PiecewiseLinearFunction]:
    """
    Split a piecewise linear function into convex upper bound segments.

    Convex parts are much easier to handle in optimization problems, thus,
    this function can be used to split a function into as few convex parts as possible.
    """
    f = minimize_piecewise_linear_function(f)
    if f.is_convex(upper_bound=upper_bound):
        return [f.model_copy(deep=True)]
    convex_parts = []
    current_segment = []
    for i, (x, y) in enumerate(zip(f.xs, f.ys, strict=True)):
        if len(current_segment) < 2:
            current_segment.append((x, y))
            continue
        previous_gradient = (current_segment[-1][1] - current_segment[-2][1]) / (
            current_segment[-1][0] - current_segment[-2][0]
        )
        current_gradient = (y - current_segment[-1][1]) / (x - current_segment[-1][0])

        def is_convex(grad1, grad2):
            return grad1 >= grad2 if upper_bound else grad1 <= grad2

        if is_convex(previous_gradient, current_gradient):
            # Still convex
            current_segment.append((x, y))
        else:
            # Not convex anymore, create a new segment
            convex_parts.append(
                PiecewiseLinearFunction(
                    xs=[x for x, _ in current_segment],
                    ys=[y for _, y in current_segment],
                )
            )
            current_segment = [current_segment[-1], (x, y)]
    if current_segment:
        convex_parts.append(
            PiecewiseLinearFunction(
                xs=[x for x, _ in current_segment], ys=[y for _, y in current_segment]
            )
        )
    return convex_parts

def split_into_segments(f: PiecewiseLinearFunction) -> List[PiecewiseLinearFunction]:
    """
    Split a piecewise linear function into segments.
    This is not a smart partitioning, but can serve as a fall-back in case of problems
    or as a baseline for experiments.
    """
    return [
        PiecewiseLinearFunction(xs=[x1, x2], ys=[y1, y2])
        for (x1, y1), (x2, y2) in f.segments()
    ]


def test_split_into_convex_upper_bound_segments():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    parts = split_into_convex_segments(f, upper_bound=True)
    assert len(parts) == 2
    assert all(p.is_convex(True) for p in parts)
    assert all(parts[0](x) == f(x) for x in range(10))
    assert all(parts[1](x) == f(x) for x in range(10, 21))
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    parts = split_into_convex_segments(f, upper_bound=True)
    assert len(parts) == 1
    assert all(p.is_convex(True) for p in parts)
    assert all(p(0) == f(0) for p in parts)
    assert all(p(10) == f(10) for p in parts)
    assert all(p(20) == f(20) for p in parts)


def generate_integer_linear_expression_from_two_points(
    x0: int, y0: int, x1: int, y1: int
) -> typing.Tuple[int, int, int]:
    """
    Generate a linear expression of the form t*y <= a*x + b where t,a,b are integers.
    It is returned as a tuple (t, a, b).
    """
    a = y1 - y0
    b = x1 - x0
    # Multiplying all y-values by this factor will make the line integer
    # for all integral x-values.
    lcm = math.lcm(abs(a), abs(b))
    assert lcm % a == 0, "There should be no rounding errors"
    y_scaling = lcm // abs(a)
    assert (
        y_scaling > 0
    ), "The scaling factor should be positive as otherwise the direction of the inequality would change"
    # The gradient by which y increases for each increase in x
    # The true gradient is gradient/y_scaling, but we want things to be integer
    gradient = (a * y_scaling) // b
    assert (a * y_scaling) % b == 0, "There should be no rounding errors"
    # The y-intercept of the line, providing us the constant term
    y_intersection = y0 * y_scaling - gradient * x0
    return (y_scaling, gradient, y_intersection)


def test_generate_integer_linear_expression():
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, 10) == (1, 1, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 20, 10) == (2, 1, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, 15) == (2, 3, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, -10) == (
        1,
        -1,
        0,
    )
    assert generate_integer_linear_expression_from_two_points(-10, -10, 10, 10) == (
        1,
        1,
        0,
    )


class PiecewiseLinearConstraint:
    """
    Create a `y <= f(x)` (upper bound) or `y >= f(x)` (lower bound) constraint for a piecewise linear function f.
    It automatically applies a convex partitioning of the function to make the constraint more efficient.
    If the function is convex, no reified constraints will be added.
    If the function is not convex, it will automatically be split into a small number of convex parts, such
    that only few decisions need to be made.

    A complexity with this constraint is the integer arithmetic. This requires automatic scaling of the
    linear constraints. Additionally, the usage of these constraints can be quite counter-intuitive, as
    you can easily create infeasible constraints, which would have been feasible with floating point arithmetic.
    Thus, this helper class also just supports lower or upper bounds, but not equality constraints.
    Still, if you end up with an infeasible model, it is quite likely that the (wrong) usage of this constraint
    is the reason.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        f: PiecewiseLinearFunction,
        upper_bound: bool,
        y_bound: Optional[int] = None,
        y: Optional[cp_model.IntVar] = None,
        add_convex_envelope: bool = True,
        optimize_convex_partition: bool = True,
    ):
        """
        Initializes a piecewise linear constraint for a constraint programming model.
        It automatically splits the function into convex parts and adds the necessary constraints.
        This can be a drastic improvement in performance for the solver, without having to
        do any reasoning on your own.

        Args:
            model (cp_model.CpModel): The constraint programming model.
            x (cp_model.IntVar): The input variable.
            f (PiecewiseLinearFunction): The piecewise linear function.
            upper_bound (bool): Flag indicating whether the constraint is an upper bound or a lower bound.
            y_bound (Optional[int], optional): The bound for the output variable y. Defaults to None, in which case the minimum or maximum value of the function is used. Only used if y is None.
            y (Optional[cp_model.IntVar], optional): The output variable y. Defaults to None, in which case a new variable is created.
            add_convex_envelope (bool, optional): If true, the convex envelope of the function is added as a constraint. Defaults to True. This can help the solver to bound the possible values of y without reified constraints, but is _theoretically_ redundant.
            optimize_convex_partition (bool, optional): If true, the function is split into as few convex parts as possible. Defaults to True. This can significantly reduce the number of reified constraints. There should be no reason to set this to false, except out of curiosity.
        Returns:
            None

        Examples:
            None
        """

        self.upper_bound = upper_bound
        # The model this constraint belongs to
        self.model = model
        # The x-variable
        self.x = x
        # The function to be a bound
        self.f = f
        # The y-variable. Either a new variable is created or an existing one is used.
        if y is None:
            # If no lower bound is provided, we use the minimum value of the function.
            # This is efficient but not necessarily what you want.
            # However, we need some lower bound to define the variable.
            if upper_bound:
                self.y = model.NewIntVar(
                    min(f.ys) if y_bound is None else y_bound, max(f.ys), "y"
                )
            else:
                self.y = model.NewIntVar(
                    min(f.ys), max(f.ys) if y_bound is None else y_bound, "y"
                )
        else:
            self.y = y
        assert isinstance(self.y, cp_model.IntVar), "y must be an integer variable"

        # restrict range of x, just to be sure it is not forgotten
        # The preprocessing will deal with this constraint easily and just
        # restrict the domain of x to the bounds of the function.
        self.model.Add(x >= f.xs[0])
        self.model.Add(x <= f.xs[-1])

        # A partition of the function into convex parts.
        # The fewer parts, the better.
        if optimize_convex_partition:
            self.convex_parts = split_into_convex_segments(f, upper_bound=upper_bound)
        else:
            # Naive splitting into segments, which is probably significantly worse
            self.convex_parts = split_into_segments(f)
        # The convex envelope of the function.
        # This is a convex function that is an upper bound of the function.
        # It can be used to help the solver, as it limits the possible values of y
        # without reified constraints, i.e., uses constraints that are always active
        # and do not need any reasoning first.
        self.envelope = get_convex_envelope(f, upper_bound=upper_bound)

        # Some statistics to evaluate how expensive the constraint is
        self.num_constraints = 0
        self.num_reified_constraints = 0
        self.num_auxiliary_variables = 0

        # If the function is convex, we can use a single constraint
        if len(self.convex_parts) == 1:
            # Very efficient constraint
            self._add_single_convex_part_constraint()
        else:
            # More complex constraint with reified constraints
            self._add_multiple_convex_parts_constraint()
            if add_convex_envelope:
                self._add_convex_envelope_constraint()

    def _add_single_convex_part_constraint(self):
        # Enforce the single convex part
        for (x1, y1), (x2, y2) in self.f.segments():
            a, b, c = generate_integer_linear_expression_from_two_points(x1, y1, x2, y2)
            if self.upper_bound:
                self.model.Add(self.y * a <= (b * self.x + c))  # type: ignore
            else:
                self.model.Add(self.y * a >= (b * self.x + c))  # type: ignore
            self.num_constraints += 1

    def _add_multiple_convex_parts_constraint(self):
        # Create boolean variables that indicate which convex part is active
        self._bvars = [
            self.model.NewBoolVar(f"b_{i}") for i in range(len(self.convex_parts))
        ]
        self.num_auxiliary_variables += len(self._bvars)
        # One of the convex parts must be active
        self.model.AddExactlyOne(self._bvars)
        self.num_constraints += 1
        for var, f in zip(self._bvars, self.convex_parts):
            # If var is true, enforce the convex constraints
            for (x1, y1), (x2, y2) in f.segments():
                a, b, c = generate_integer_linear_expression_from_two_points(
                    x1, y1, x2, y2
                )
                if self.upper_bound:  # <= if upper_bound, >= if lower_bound
                    self.model.Add((self.y * a) <= (b * self.x + c)).OnlyEnforceIf(var)  # type: ignore
                else:
                    self.model.Add((self.y * a) >= (b * self.x + c)).OnlyEnforceIf(var)  # type: ignore
                # x must be within the bounds of the convex part
                self.model.Add(self.x <= f.get_bounds()[1]).OnlyEnforceIf(var)
                self.model.Add(self.x >= f.get_bounds()[0]).OnlyEnforceIf(var)
                # We could save two constraints here, but they won't make much of a difference
                # The bespoken two constraints are the left-most and right-most x-values of
                # the first and the last convex part.
                self.num_constraints += 3
                self.num_reified_constraints += 3

    def _add_convex_envelope_constraint(self):
        """
        Add the convex envelope as a constraint to strengthen the model
        This constraint is redundant but can help the solver
        """
        for (x1, y1), (x2, y2) in self.envelope.segments():
            a, b, c = generate_integer_linear_expression_from_two_points(x1, y1, x2, y2)
            if self.upper_bound:
                self.model.Add(a * self.y <= (b * self.x + c))  # type: ignore
            else:
                self.model.Add(a * self.y >= (b * self.x + c))  # type: ignore
            self.num_constraints += 1


def test_piecewise_linear_upper_bound_constraint():
    model = cp_model.CpModel()
    x = model.NewIntVar(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.Maximize(c.y)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(c.y) == 10
    assert solver.Value(x) == 10
    assert c.num_auxiliary_variables == 0
    assert c.num_constraints == 2

    model = cp_model.CpModel()
    x = model.NewIntVar(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.Maximize(c.y)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(c.y) == 50
    assert solver.Value(x) == 20
    assert c.num_auxiliary_variables == 2

    model = cp_model.CpModel()
    x = model.NewIntVar(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=False)
    model.Minimize(c.y)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(c.y) == 0
    assert solver.Value(x) == 0

    model = cp_model.CpModel()
    x = model.NewIntVar(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20, 30], ys=[20, 10, 50, 40])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=False)
    model.Minimize(c.y)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(c.y) == 10
    assert solver.Value(x) == 10
