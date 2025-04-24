from ortools.sat.python import cp_model


class ExpectFeasible:
    """
    A context manager for testing model feasibility.
    It will yield a model and automatically run the solver, expecting it to be feasible.
    Otherwise, it will raise an exception.
    """

    def __init__(
        self,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = cp_model.CpModel() if model is None else model
        self.solver = cp_model.CpSolver() if solver is None else solver

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val)
        status = self.solver.Solve(self.model)
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            raise RuntimeError(
                "Model is not feasible, but we expected it to be feasible."
            )


class ExpectInfeasible:
    """
    A context manager for testing model infeasibility.
    It will yield a model and automatically run the solver, expecting it to be infeasible.
    Otherwise, it will raise an exception.
    """

    def __init__(
        self,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = cp_model.CpModel() if model is None else model
        self.solver = cp_model.CpSolver() if solver is None else solver

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val)
        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            raise RuntimeError(
                "Model is feasible, but we expected it to be infeasible."
            )


class ExpectObjective:
    """
    A context manager for testing model feasibility.
    It will yield a model and automatically run the solver, expecting it to be feasible.
    Otherwise, it will raise an exception.
    """

    def __init__(
        self,
        objective: float,
        tol: float = 1e-10,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = cp_model.CpModel() if model is None else model
        self.solver = cp_model.CpSolver() if solver is None else solver
        self.objective = objective
        self.tol = tol

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val)
        status = self.solver.solve(self.model)
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            raise RuntimeError(
                "Model is not feasible, but we expected it to be feasible."
            )
        if abs(self.solver.ObjectiveValue() - self.objective) > self.tol:
            raise RuntimeError(
                f"Model objective is {self.solver.ObjectiveValue()}, but we expected it to be {self.objective}."
            )


class ExpectOptimal:
    """
    A context manager for testing model optimality with a certain time limit.
    It essentially verifies that the model is easy to solve.
    """

    def __init__(
        self,
        time_limit: float = 1.0,
        model: cp_model.CpModel | None = None,
        solver: cp_model.CpSolver | None = None,
    ):
        self.model = cp_model.CpModel() if model is None else model
        self.solver = cp_model.CpSolver() if solver is None else solver
        self.time_limit = time_limit

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val)
        self.solver.parameters.max_time_in_seconds = self.time_limit
        status = self.solver.Solve(self.model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "Model is not optimal, but we expected it to be optimal."
            )
