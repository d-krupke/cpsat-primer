from ortools.sat.python import cp_model


class ExpectFeasible:
    """
    Context manager that asserts a CP-SAT model is feasible.

    Usage:
        with ExpectFeasible() as model:
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
        status = self.solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Expected feasible, but solver returned status {status}."
            )


class ExpectInfeasible:
    """
    Context manager that asserts a CP-SAT model is infeasible.

    Usage:
        with ExpectInfeasible() as model:
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
        status = self.solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Expected infeasible, but solver returned status {status}."
            )


class ExpectObjective:
    """
    Context manager that asserts a CP-SAT model's objective value.

    Usage:
        with ExpectObjective(target, tol=1e-6) as model:
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
        value = self.solver.ObjectiveValue()
        if abs(value - self.expected) > self.tol:
            raise RuntimeError(
                f"Objective {value} differs from expected {self.expected} by more than {self.tol}."
            )


class ExpectOptimal:
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
        status = self.solver.Solve(self.model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                f"Expected optimal within {self.time_limit}s, but status {status}."
            )


def solve(
    model,
    solver: cp_model.CpSolver | None = None,
    expect: int = cp_model.OPTIMAL,
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
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    assert status == expect, f"Expected status={expect}, got {status}"
    return solver
