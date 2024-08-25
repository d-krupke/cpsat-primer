"""
This file implements a piecewise constant function.
It create a variable y that is constrained to be piecewise constant
with respect to x, i.e., y = f(x) where f is a piecewise constant function.

The function is defined by a pydantic model, allowing for easy serialization and validation.
This is useful if you want to define your function in a config file.

The code is under MIT license, and is free to use, modify, or distribute.
Just copy and paste for whatever project you are working on.

https://github.com/d-krupke/cpsat-primer

Author: Dominik Krupke (2024)
"""

from typing import List
from ortools.sat.python import cp_model
import bisect
from pydantic import BaseModel, model_validator


class PiecewiseConstantFunction(BaseModel):
    """
    Defines a piecewise constant function, also known as a step function.
    The function is defined by a list of x values and a list of y values.
    The function is constant between the x values.
    The function is defined for x in [xs[0], xs[-1]).
    For xs[i] it takes the value ys[i], thus, the last value in xs defines the beginning of
    the undefined interval.

    Example:
    ```python
    f = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 2])
    assert f(2) == 2
    assert f(0) == 0
    assert f(1) == 1
    assert f.is_monotonous()
    ```

    """

    xs: List[int]
    ys: List[int]

    def __call__(self, x: int) -> int:
        """
        Returns the value of the function at x.
        """
        if not self.is_defined_for(x):
            raise ValueError(f"x={x} is out of bounds")
        # binary search for the interval
        i = bisect.bisect_right(self.xs, x) - 1
        return self.ys[i]

    def is_monotonous(self) -> bool:
        """
        Returns True if the function is monotonous.
        Monotonous functions are usually easier to optimize.
        """
        return all(y1 <= y2 for y1, y2 in zip(self.ys[:-1], self.ys[1:])) or all(
            y1 >= y2 for y1, y2 in zip(self.ys[:-1], self.ys[1:])
        )

    def is_defined_for(self, x: int) -> bool:
        return self.xs[0] <= x < self.xs[-1]

    @model_validator(mode="after")
    def _check_data(cls, v):
        if len(v.xs) != len(v.ys) + 1:
            raise ValueError("len(xs) must be len(ys) + 1")
        if any(x1 >= x2 for x1, x2 in zip(v.xs[:-1], v.xs[1:])):
            raise ValueError("xs must be strictly increasing")
        return v


class PiecewiseConstantConstraint:
    """
    This class creates a constraint that enforces y = f(x) where f is a piecewise constant function.
    This is implemented by creating a boolean variable for each step in the function and adding
    the difference between the steps to y. This can be very efficient for monotonous functions
    as all the coefficients would have the same sign.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        x_var: cp_model.IntVar,
        f: PiecewiseConstantFunction,
        restrict_domain: bool = False,
    ):
        self.x = x_var
        self.f = f
        self.model = model

        if restrict_domain:
            self.y_domain = cp_model.Domain.from_values(list(set(f.ys)))
            self.y = self.model.new_int_var_from_domain(self.y_domain, "y")
        else:
            self.y = self.model.new_int_var(min(f.ys), max(f.ys), "y")
        # Limit the range of x.
        self.model.add(self.x >= self.f.xs[0])
        self.model.add(self.x <= self.f.xs[-1] - 1)
        # Create a boolean variable for each step
        # The i-th step will force y=ys[i].
        # If no step is made, y=ys[0].
        self._step_var = [
            self.model.new_bool_var(f"interval_{i}") for i in range(len(self.f.ys) - 1)
        ]
        self._enforce_step_order()

        # Match y to the corresponding interval.
        self.y_expr = self.f.ys[0] + sum(
            step * (self.f.ys[i + 1] - self.f.ys[i])
            for i, step in enumerate(self._step_var)
        )
        self.model.add(self.y == self.y_expr)
        self._enforce_steps_for_x()

    def _enforce_step_order(self):
        # Enforce that the second step is only made if the first step is made
        for i in range(len(self._step_var) - 1):
            # All previous steps need to be made, too.
            self.model.add(self._step_var[i] >= self._step_var[i + 1])

    def _enforce_steps_for_x(self):
        # Enforce that exactly the right number of steps is made
        # depending on x.
        # If no step is made: x >= xs[0]
        # If one step is made x >= xs[0] + (xs[1] - xs[0]) = xs[1]
        # ...
        self.model.add(
            self.x
            >= self.f.xs[0]
            + sum(
                step * (self.f.xs[i + 1] - self.f.xs[i])
                for i, step in enumerate(self._step_var)
            )  # type: ignore
        )
        # If no step is made: x+1 <= self.xs[1]
        # If one step is made x+1 <= xs[1] + (xs[2] - xs[1]) = xs[2]
        # ...
        self.model.add(
            self.x + 1
            <= self.f.xs[1]
            + sum(
                step * (self.f.xs[i + 2] - self.f.xs[i + 1])
                for i, step in enumerate(self._step_var)
            )  # type: ignore
        )

    def __call__(self, x: int) -> int:
        """
        Returns the value of the function at x.
        This is not a constraint but primarily for testing/logging.
        """
        return self.f(x)

    def is_monotonous(self) -> bool:
        """
        Returns True if the function is monotonous.
        Monotonous functions are usually easier to optimize.
        """
        return self.f.is_monotonous()


def test_stairs():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 2])
    f = PiecewiseConstantConstraint(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 2
    assert solver.value(f.y) == 2
    assert f(2) == 2
    assert f(0) == 0
    assert f(1) == 1
    assert f.is_monotonous()


def test_stairs_min():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 2])
    f = PiecewiseConstantConstraint(
        model,
        x,
        f_,
    )
    model.minimize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 0
    assert solver.value(f.y) == 0
    assert f(0) == 0
    assert f.is_monotonous()


def test_pyramid():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 0])
    f = PiecewiseConstantConstraint(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 1
    assert solver.value(f.y) == 1
    assert f(1) == 1
    assert not f.is_monotonous()


def test_larger_pyramid():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3, 4, 5], ys=[0, 1, 5, 1, 0])
    f = PiecewiseConstantConstraint(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 2
    assert solver.value(f.y) == 5
    assert f(2) == 5
    assert f(3) == 1
    assert not f.is_monotonous()


class PiecewiseConstantConstraintViaOnlyIf:
    """
    This class creates a constraint that enforces y = f(x) where f is a piecewise constant function.
    This is implemented by via OnlyIf constraints, which is probably less efficient.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        x_var: cp_model.IntVar,
        f: PiecewiseConstantFunction,
    ):
        self.x = x_var
        self.f = f
        self.model = model
        self.y_domain = cp_model.Domain.from_values(list(set(f.ys)))
        self.y = model.new_int_var_from_domain(self.y_domain, "y")
        # Limit the range of x.
        self.model.add(self.x >= self.f.xs[0])
        self.model.add(self.x <= self.f.xs[-1] - 1)
        # Create a boolean variable for each interval
        ivars = [model.new_bool_var(f"interval_{i}") for i in range(len(self.f.ys))]
        # Enforce that exactly one interval is active
        model.add_exactly_one(ivars)
        # Match y to the corresponding interval.
        for i, ivar in enumerate(ivars):
            model.add(self.y == self.f.ys[i]).only_enforce_if(ivar)
            model.add(self.x >= self.f.xs[i]).only_enforce_if(ivar)
            model.add(self.x <= self.f.xs[i + 1] - 1).only_enforce_if(ivar)

    def __call__(self, x: int) -> int:
        """
        Returns the value of the function at x.
        This is not a constraint but primarily for testing/logging.
        """
        return self.f(x)

    def is_monotonous(self) -> bool:
        """
        Returns True if the function is monotonous.
        Monotonous functions are usually easier to optimize.
        """
        return self.f.is_monotonous()


def test_stairs_onlyif():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 2])
    f = PiecewiseConstantConstraintViaOnlyIf(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 2
    assert solver.value(f.y) == 2
    assert f(2) == 2
    assert f(0) == 0
    assert f(1) == 1
    assert f.is_monotonous()


def test_stairs_min_onlyif():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 2])
    f = PiecewiseConstantConstraintViaOnlyIf(
        model,
        x,
        f_,
    )
    model.minimize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 0
    assert solver.value(f.y) == 0
    assert f(0) == 0
    assert f.is_monotonous()


def test_pyramid_onlyif():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3], ys=[0, 1, 0])
    f = PiecewiseConstantConstraintViaOnlyIf(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 1
    assert solver.value(f.y) == 1
    assert f(1) == 1
    assert not f.is_monotonous()


def test_larger_pyramid_onlyif():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f_ = PiecewiseConstantFunction(xs=[0, 1, 2, 3, 4, 5], ys=[0, 1, 5, 1, 0])
    f = PiecewiseConstantConstraintViaOnlyIf(
        model,
        x,
        f_,
    )
    model.maximize(f.y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(x) == 2
    assert solver.value(f.y) == 5
    assert f(2) == 5
    assert f(3) == 1
    assert not f.is_monotonous()
