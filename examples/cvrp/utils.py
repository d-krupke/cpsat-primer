"""
This file provides a set of context managers and utility functions for working with
Google OR-Tools CP-SAT solver. These context managers help to assert the feasibility,
infeasibility, optimality, and objective value of CP-SAT models during testing.

It is especially intended to test for the trivial cases of your models, where you
can still can quickly get an off by one or wrong sign bugs. In some cases, these
may even remain unnoticed for a while.

Using these utils, you can super quickly write these trivial test cases and safe
yourself some time.
```python
def test_example():
    with expect_feasible() as model:
        # build model constraints
        x = model.new_bool_var("x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y == 1)
        # This will raise a RuntimeError if the model is infeasible.
```

Alternatively, you can use `assert_feasible()` and similar functions to check
the status of a model after solving it. These functions are useful for
asserting the status of a model without using context managers.
```python
def test_example():
    model = cp_model.CpModel()
    x = model.new_bool_var("x")
    y = model.new_int_var(0, 10, "y")
    model.add(x + y == 1)
    # This will raise a RuntimeError if the model is infeasible.
    assert_feasible(model)
```
"""

from ortools.sat.python import cp_model


class ExpectModelFeasible:
    """
    Context manager that asserts a CP-SAT model is feasible.

    Usage:
        with ExpectModelFeasible() as model:
            # build model constraints
            x = model.new_bool_var("x")
            y = model.new_int_var(0, 10, "y")
            model.add(x + y == 1)
            # This will raise a RuntimeError if the model is infeasible.

    Args:
        model (cp_model.CpModel | None): An existing CP-SAT model or None to create a new one.
        solver (cp_model.CpSolver | None): An existing solver or None to create a new one.

    Raises:
        RuntimeError: If the model status is neither OPTIMAL nor FEASIBLE.
    """

    def __init__(
        self,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = model or cp_model.CpModel()
        self.solver = solver or cp_model.CpSolver()

    def __enter__(self) -> cp_model.CpModel:
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            # Propagate exceptions raised inside the with-block
            raise exc_type(exc_val)
        status = self.solver.solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Expected feasible, but solver returned status {status}."
            )


class ExpectModelInfeasible:
    """
    Context manager that asserts a CP-SAT model is infeasible.

    Usage:
        with ExpectModelInfeasible() as model:
            # build model constraints that cannot all be satisfied
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.add(x + y == 1)
            model.add(x + y == 2)
            # This will raise a RuntimeError if the model is feasible.

    Args:
        model (cp_model.CpModel | None): An existing CP-SAT model or None to create a new one.
        solver (cp_model.CpSolver | None): An existing solver or None to create a new one.

    Raises:
        RuntimeError: If the model is found to be feasible.
    """

    def __init__(
        self,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = model or cp_model.CpModel()
        self.solver = solver or cp_model.CpSolver()

    def __enter__(self) -> cp_model.CpModel:
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            raise exc_type(exc_val)
        status = self.solver.solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Expected infeasible, but solver returned status {status}."
            )


class ExpectObjectiveValue:
    """
    Context manager that asserts a CP-SAT model's objective value.

    Usage:
        with ExpectObjectiveValue(target, tol=1e-6) as model:
            # build model, add minimize() or maximize()
            pass

    Args:
        objective (float): Expected objective value.
        tol (float): Acceptable absolute error tolerance.
        model (cp_model.CpModel | None): Existing model or None.
        solver (cp_model.CpSolver | None): Existing solver or None.

    Raises:
        RuntimeError: If the model is infeasible or objective differs by more than tol.
    """

    def __init__(
        self,
        objective: float,
        tol: float = 1e-10,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.expected = objective
        self.tol = tol
        self.model = model or cp_model.CpModel()
        self.solver = solver or cp_model.CpSolver()

    def __enter__(self) -> cp_model.CpModel:
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            raise exc_type(exc_val)
        status = self.solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Expected feasible for objective check, but status {status}."
            )
        value = self.solver.objective_value
        if abs(value - self.expected) > self.tol:
            raise RuntimeError(
                f"Objective {value} differs from expected {self.expected} by more than {self.tol}."
            )


class ExpectOptimalWithinTime:
    """
    Context manager that asserts a CP-SAT model solves to optimal within a time limit.

    Usage:
        with ExpectOptimal(time_limit=2.0) as model:
            # build model and set objective
            pass

    Args:
        time_limit (float): Maximum solve time in seconds.
        model (cp_model.CpModel | None): Existing model or None.
        solver (cp_model.CpSolver | None): Existing solver or None.

    Raises:
        RuntimeError: If the solver does not reach OPTIMAL status.
    """

    def __init__(
        self,
        time_limit: float = 1.0,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.time_limit = time_limit
        self.model = model or cp_model.CpModel()
        self.solver = solver or cp_model.CpSolver()

    def __enter__(self) -> cp_model.CpModel:
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            raise exc_type(exc_val)
        self.solver.parameters.max_time_in_seconds = self.time_limit
        status = self.solver.solve(self.model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                f"Expected optimal within {self.time_limit}s, but status {status}."
            )


def solve(
    model,
    solver: cp_model.CpSolver | None = None,
    expect: int | list[int] = cp_model.OPTIMAL,
    time_limit: float | None = None,
) -> cp_model.CpSolver:
    """
    Solve a CP-SAT model and assert its status.

    Args:
      model: any object with a `.model` attribute (a CpModel).
      solver:      optional CpSolver to use (new one if None).
      expect:      expected status code (e.g. OPTIMAL or FEASIBLE).
      time_limit:  optional max_time_in_seconds.

    Returns:
      The solver after solving.

    Raises:
      AssertionError if the solver’s status != expect.
    """
    solver = cp_model.CpSolver() if solver is None else solver
    if isinstance(expect, int):
        expect = [expect]
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    assert status in expect, f"Expected status in {expect}, got {status}"
    return solver


def _solve(model: cp_model.CpModel, solver=None, time_limit=None):
    solver = solver or cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    return solver, status


def assert_feasible(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver | None = None,
    time_limit: float | None = None,
):
    """
    Solve `model` and assert status is OPTIMAL or FEASIBLE.
    Returns the solver for further inspection if needed.
    """
    solver, status = _solve(model, solver, time_limit)
    assert status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ), f"Expected feasible or optimal, got {status}"
    return solver


def assert_infeasible(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver | None = None,
    time_limit: float | None = None,
):
    """
    Solve `model` and assert status is INFEASIBLE.
    """
    _, status = _solve(model, solver, time_limit)
    assert status == cp_model.INFEASIBLE, f"Expected infeasible, got {status}"


def assert_optimal(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver | None = None,
    time_limit: float | None = None,
):
    """
    Solve `model` and assert status is OPTIMAL.
    """
    _, status = _solve(model, solver, time_limit)
    assert status == cp_model.OPTIMAL, f"Expected optimal, got {status}"


def assert_objective(
    model: cp_model.CpModel,
    expected: float,
    tol: float = 1e-8,
    solver: cp_model.CpSolver | None = None,
    time_limit: float | None = None,
):
    """
    Solve `model`, assert it's feasible or optimal, then
    check |ObjectiveValue - expected| <= tol.
    """
    solver, status = _solve(model, solver, time_limit)
    assert status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ), f"Expected feasible or optimal, got {status}"
    val = solver.objective_value
    assert abs(val - expected) <= tol, f"Expected objective≈{expected}, got {val}"
    return solver
