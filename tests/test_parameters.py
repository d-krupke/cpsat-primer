import random
from ortools.sat.python import cp_model


def test_basic_parameters_for_attribute_errors():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.relative_gap_limit = 0.05
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("We have a solution.")
    else:
        print("Help?! No solution available! :( ")

    status_name = solver.status_name(status)  # noqa: F841
    bound = solver.best_objective_bound  # noqa: F841


def test_callback():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 100, "x")

    class MySolutionCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, stuff):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.stuff = (
                stuff  # just to show that we can save some data in the callback.
            )

        def on_solution_callback(self):
            obj = self.objective_value  # best solution value
            bound = self.best_objective_bound  # best bound
            print(f"The current value of x is {self.value(x)}")
            if abs(obj - bound) < 10:
                self.StopSearch()  # abort search for better solution
            # ...

    solver = cp_model.CpSolver()
    solver.solve(model, MySolutionCallback(None))


def test_further_properties():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    status = solver.solve(model)  # noqa: F841
    solver.num_booleans
    solver.num_branches
    solver.num_conflicts


def test_bound_callback():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 100, "x")  # noqa: F841

    solver = cp_model.CpSolver()

    def bound_callback(bound):
        print(f"New bound: {bound}")
        if bound > 100:
            solver.stop_search()

    solver.best_bound_callback = bound_callback
    status = solver.solve(model)  # noqa: F841


def test_callbacks_via_tsp():
    # create a random tsp problem so we have something to solve that
    # cannot be resolved in the preprocessing phase
    model = cp_model.CpModel()
    vertices = range(80)
    arcs = [(i, j) for i in vertices for j in vertices if i != j]
    costs = {(i, j): random.randint(1, 100) for i, j in arcs}
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in arcs}
    model.add_circuit([(v, w, x) for (v, w), x in x.items()])
    model.minimize(sum(costs[i, j] * x[i, j] for i, j in arcs))
    solver = cp_model.CpSolver()

    def bound_callback(bound):
        print(f"New bound: {bound}")
        if bound > 200:
            print("Abort search due to bound")
            solver.stop_search()

    solver.best_bound_callback = bound_callback
    status = solver.solve(model)
    print(solver.status_name(status))


def test_callbacks_via_tsp_callable():
    # create a random tsp problem so we have something to solve that
    # cannot be resolved in the preprocessing phase
    model = cp_model.CpModel()
    vertices = range(80)
    arcs = [(i, j) for i in vertices for j in vertices if i != j]
    costs = {(i, j): random.randint(1, 100) for i, j in arcs}
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in arcs}
    model.add_circuit([(v, w, x) for (v, w), x in x.items()])
    model.minimize(sum(costs[i, j] * x[i, j] for i, j in arcs))
    solver = cp_model.CpSolver()

    class BoundCallback:
        def __init__(self, solver) -> None:
            self.solver = solver

        def __call__(self, bound):
            print(f"New bound: {bound}")
            if bound > 200:
                print("Abort search due to bound")
                self.solver.stop_search()

    solver.best_bound_callback = BoundCallback(solver)
    status = solver.solve(model)
    print(solver.status_name(status))


def test_assumptions():
    model = cp_model.CpModel()
    b1 = model.new_bool_var("b1")
    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    model.add_assumptions([b1, ~b2])  # assume b1=True, b2=False
    model.add_assumption(b3)  # assume b3=True (single literal)
    # ... solve again and analyze ...
    model.clear_assumptions()  # clear all assumptions


def test_bad_hints():
    """
    Test that fix_variables_to_their_hinted_value detects infeasible hints.
    This is more reliable than debug_crash_on_bad_hint which has race conditions.
    """
    model = cp_model.CpModel()
    vertices = range(10)
    arcs = [(i, j) for i in vertices for j in vertices if i != j]
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in arcs}
    model.add_circuit([(v, w, x) for (v, w), x in x.items()])
    # add a bad hint of multiple outgoing arcs from the same node (violates circuit)
    model.add_hint(x[0, 1], 1)
    model.add_hint(x[0, 2], 1)
    model.add_hint(x[0, 3], 1)
    solver = cp_model.CpSolver()
    # Fix variables to hinted values to reliably detect infeasible hints
    solver.parameters.fix_variables_to_their_hinted_value = True
    status = solver.solve(model)
    assert status == cp_model.INFEASIBLE, "Expected INFEASIBLE due to conflicting hints"


def test_presolve_parameters_exist():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.cp_model_presolve = False
    solver.parameters.max_presolve_iterations = 3
    solver.parameters.cp_model_probing_level = 1
    solver.parameters.presolve_probing_deterministic_time_limit = 5
