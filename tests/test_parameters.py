"""Check solver-parameter behavior the parameters chapter describes.

The chapter's parameter snippets are smoke-tested via their markers; these
tests cover behavior that needs a harder model or extra assertions.
"""

import random

from _snippets import run_snippet
from ortools.sat.python import cp_model


def test_further_properties():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert min(solver.num_booleans, solver.num_branches, solver.num_conflicts) >= 0


def random_tsp(n: int = 80) -> cp_model.CpModel:
    """A TSP that presolve cannot resolve; its optimum (906) is far above the
    callbacks' thresholds, so they have to stop the search."""
    rng = random.Random(0)
    model = cp_model.CpModel()
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i, j in arcs}
    model.add_circuit([(v, w, var) for (v, w), var in x.items()])
    model.minimize(sum(rng.randint(10, 100) * x[arc] for arc in arcs))
    return model


def test_bound_callback_stops_search(capsys):
    """The chapter's function callback sees the bound pass 100 and stops."""
    solver = run_snippet("parameters/bound_callback")["solver"]
    solver.parameters.max_time_in_seconds = 30
    assert solver.solve(random_tsp()) in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    assert solver.best_objective_bound > 100
    assert "New bound:" in capsys.readouterr().out


def test_bound_callback_class_stops_search(capsys):
    """The callable class works the same way, here with a threshold of 200."""
    bound_callback = run_snippet("parameters/bound_callback_class")["BoundCallback"]
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.best_bound_callback = bound_callback(solver)
    assert solver.solve(random_tsp()) in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    assert "Abort search due to bound" in capsys.readouterr().out


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
