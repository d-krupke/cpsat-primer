"""MathOpt examples beyond the chapter's snippets.

The chapter's own examples (the step-by-step walkthrough, the Stigler diet
and set cover) are tested via their snippet markers in
`test_documented_snippets.py`.
"""

import math
from datetime import timedelta

from ortools.math_opt.python import mathopt as mo


def test_mathopt():
    """Small, self-contained MIP with MathOpt showcasing variables, constraints, objective, and solve params."""
    # --- 1) Build the model ---------------------------------------------------
    model = mo.Model(name="tiny_mip")

    # Decision variables
    x = model.add_binary_variable(name="x")  # x ∈ {0,1}
    y = model.add_integer_variable(lb=0, ub=10, name="y")  # bounded integer
    z = model.add_variable(lb=-math.inf, ub=12.3, name="z")  # continuous (unbounded)

    # --- 2) Add constraints ---------------------------------------------------
    # Simple linear constraint: x + 2y + 3z ≤ 4
    model.add_linear_constraint(x + 2 * y + 3.5 * z <= 4.1, name="c1_basic")

    # For longer expressions, prefer fast_sum over Python's sum for speed.
    # Tip: start=1 so we don't accidentally get a zero coefficient on x.
    model.add_linear_constraint(
        mo.fast_sum(coef * var for coef, var in enumerate([x, y, z], start=1)) <= 5,
        name="c2_fast_sum",
    )

    # You can give both lower and upper bounds in one linear constraint:
    # 1 ≤ 3x + 2y + z ≤ 10
    model.add_linear_constraint(lb=1, expr=3 * x + 2.5 * y + z, ub=10, name="c3_box")

    # --- 3) Objective: maximize a linear expression ---------------------------
    model.set_objective(
        mo.fast_sum(coef * var for coef, var in enumerate([x, y, z], start=1)),
        is_maximize=True,
    )

    # --- 4) Solve parameters ---------------------------------------------------
    params = mo.SolveParameters(
        time_limit=timedelta(seconds=30),  # wall-clock time limit
        relative_gap_tolerance=0.01,  # 1% MIP gap
        absolute_gap_tolerance=0.01,  # absolute gap
        enable_output=True,  # solver logs to stdout
    )

    # Choose a solver supported by MathOpt (HiGHS here).
    result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)

    # --- 5) Inspect the result -------------------------------------------------
    # It’s good practice to look at termination info & basic stats.
    term = result.termination
    print(f"Termination: {term.reason.name}")
    if term.detail:
        print(f"Detail: {term.detail}")

    if hasattr(result, "solve_stats") and result.solve_stats is not None:
        stats = result.solve_stats
        # Not all fields are available across solvers; print what we can.
        print("Solve stats:", stats)

    if result.has_primal_feasible_solution():
        print("\nObjective value:", result.objective_value())

        # Option A: get the full map (works for many vars, then index by handle)
        values = result.variable_values()
        print("Variable values (via map):")
        print("  x:", values.get(x))
        print("  y:", values.get(y))
        print("  z:", values.get(z))

        # Option B: request a list in the same order you pass in
        x_val, y_val, z_val = result.variable_values([x, y, z])
        print("Variable values (via list):")
        print(f"  x: {x_val}, y: {y_val}, z: {z_val}")
    else:
        print("No primal feasible solution found.")


if __name__ == "__main__":
    test_mathopt()
