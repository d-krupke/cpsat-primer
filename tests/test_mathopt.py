from datetime import timedelta
import math
from ortools.math_opt.python import mathopt as mo
import networkx as nx
import matplotlib.pyplot as plt


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


def test_mathopt_stigler_diet():
    """
    Simplified Stigler Diet:
    - Choose nonnegative servings of a few foods
    - Minimize total cost
    - Satisfy minimum requirements for Calories, Protein, Calcium
    Units are illustrative only (don’t take the data as nutritional truth).
    """
    # --- Data -----------------------------------------------------------------
    foods = ["Wheat Flour", "Milk", "Cabbage", "Beef"]
    # Cost per serving (EUR)
    cost = {
        "Wheat Flour": 0.36,
        "Milk": 0.23,
        "Cabbage": 0.10,
        "Beef": 1.20,
    }

    # Nutrients per serving (approximate / illustrative)
    #               Calories  Protein(g)  Calcium(mg)
    calories = {"Wheat Flour": 364.0, "Milk": 150.0, "Cabbage": 25.0, "Beef": 250.0}
    protein = {"Wheat Flour": 10.0, "Milk": 8.0, "Cabbage": 1.3, "Beef": 26.0}
    calcium = {"Wheat Flour": 15.0, "Milk": 285.0, "Cabbage": 40.0, "Beef": 20.0}

    # Requirements (minimums)
    req = {
        "Calories": 2000.0,  # kcal
        "Protein": 55.0,  # grams
        "Calcium": 800.0,  # mg
    }

    # --- Model ----------------------------------------------------------------
    model = mo.Model(name="stigler_diet")

    # Decision: servings of each food (continuous, ≥ 0). You could switch to integers to show a MIP.
    servings = {f: model.add_variable(lb=0.0, name=f"servings[{f}]") for f in foods}

    # Optionally cap servings to keep things tidy (purely illustrative)
    for f in foods:
        model.add_linear_constraint(servings[f] <= 20.0, name=f"cap[{f}]")

    # Nutrient constraints ------------------------------------------------------
    # Calories
    model.add_linear_constraint(
        mo.fast_sum(calories[f] * servings[f] for f in foods) >= req["Calories"],
        name="nutrients[Calories]",
    )
    # Protein
    model.add_linear_constraint(
        mo.fast_sum(protein[f] * servings[f] for f in foods) >= req["Protein"],
        name="nutrients[Protein]",
    )
    # Calcium
    model.add_linear_constraint(
        mo.fast_sum(calcium[f] * servings[f] for f in foods) >= req["Calcium"],
        name="nutrients[Calcium]",
    )

    # Objective: minimize total cost -------------------------------------------
    model.set_objective(
        mo.fast_sum(cost[f] * servings[f] for f in foods), is_maximize=False
    )

    # Solve parameters ----------------------------------------------------------
    params = mo.SolveParameters(
        time_limit=timedelta(seconds=10),
        relative_gap_tolerance=1e-6,  # LP, so we can be tight
        enable_output=False,
    )

    result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)

    # --- Report ----------------------------------------------------------------
    term = result.termination
    print(f"Termination: {term.reason.name}")

    if not result.has_primal_feasible_solution():
        print("No feasible solution found.")
        return

    print(f"Min cost: €{result.objective_value():.2f}")

    # Extract chosen servings (suppress near-zeros)
    values = result.variable_values()
    print("\nServings:")
    for f in foods:
        qty = values.get(servings[f], 0.0)
        if abs(qty) > 1e-9:
            print(f"  {f:12s}: {qty:8.3f}")

    # Compute realized totals
    tot_cal = sum(calories[f] * values.get(servings[f], 0.0) for f in foods)
    tot_pro = sum(protein[f] * values.get(servings[f], 0.0) for f in foods)
    tot_calcium = sum(calcium[f] * values.get(servings[f], 0.0) for f in foods)

    print("\nNutrient totals (min required in parentheses):")
    print(f"  Calories: {tot_cal:.1f}  ({req['Calories']})")
    print(f"  Protein : {tot_pro:.1f}  ({req['Protein']})")
    print(f"  Calcium : {tot_calcium:.1f}  ({req['Calcium']})")


def solve_fixed_charge_flow():
    """
    Fixed-charge network flow (binary arc activation + continuous flow).
    - We must send D units from source 's' to sink 't'.
    - Using an arc incurs:
        * a fixed cost if it's activated (binary y_e = 1)
        * a variable cost per unit of flow (continuous f_e >= 0)
    - Linking constraint: 0 <= f_e <= capacity_e * y_e   (big-M with M = capacity)
    """

    # ----------------- Build a small directed graph with data -----------------
    G = nx.DiGraph()
    # Nodes
    s, t = "s", "t"
    nodes = [s, "a", "b", "c", t]
    G.add_nodes_from(nodes)

    # (u, v, capacity, variable_cost, fixed_cost)
    arcs = [
        (s, "a", 6.3, 0.18, 0.30),
        (s, "b", 5.7, 0.17, 0.35),
        ("a", "c", 4.8, 0.10, 0.25),
        ("b", "c", 5.2, 0.11, 0.25),
        ("a", t, 3.1, 0.28, 0.30),
        ("b", t, 2.9, 0.30, 0.30),
        ("c", t, 8.4, 0.06, 0.45),
    ]
    for u, v, cap, vc, fc in arcs:
        G.add_edge(u, v, capacity=cap, vc=vc, fc=fc)
    D = 10.5  # demand to ship from s to t

    # --------------------------- Build MathOpt model ---------------------------
    model = mo.Model(name="fixed_charge_flow")

    # Flow and activation variables per arc
    flow = {}
    use = {}
    for u, v in G.edges:
        cap = G[u][v]["capacity"]
        flow[(u, v)] = model.add_variable(lb=0.0, ub=cap, name=f"f[{u}->{v}]")
        use[(u, v)] = model.add_binary_variable(name=f"y[{u}->{v}]")
        # Link flow to activation: f_uv ≤ cap * y_uv
        model.add_linear_constraint(
            flow[(u, v)] <= cap * use[(u, v)], name=f"link[{u}->{v}]"
        )

    # Flow conservation
    for node in G.nodes:
        inflow = mo.fast_sum(
            flow[(u, node)] for u in G.predecessors(node) if (u, node) in flow
        )
        outflow = mo.fast_sum(
            flow[(node, v)] for v in G.successors(node) if (node, v) in flow
        )
        if node == s:
            # net outflow = D
            model.add_linear_constraint(outflow - inflow == D, name=f"balance[{node}]")
        elif node == t:
            # net inflow = D  -> out - in = -D
            model.add_linear_constraint(outflow - inflow == -D, name=f"balance[{node}]")
        else:
            model.add_linear_constraint(
                outflow - inflow == 0.0, name=f"balance[{node}]"
            )

    # Objective: minimize variable + fixed costs
    var_cost_term = mo.fast_sum(G[u][v]["vc"] * flow[(u, v)] for (u, v) in G.edges)
    fix_cost_term = mo.fast_sum(G[u][v]["fc"] * use[(u, v)] for (u, v) in G.edges)
    model.set_objective(var_cost_term + fix_cost_term, is_maximize=False)

    # ------------------------------ Solve -------------------------------------
    params = mo.SolveParameters(
        time_limit=timedelta(seconds=15),
        relative_gap_tolerance=1e-6,
        enable_output=False,
    )
    # If your build supports it, HIGHS handles MIPs. Otherwise switch to GSCIP.
    result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)

    print(f"Termination: {result.termination.reason.name}")
    if not result.has_primal_feasible_solution():
        print("No feasible solution.")
        return

    values = result.variable_values()
    total_cost = result.objective_value()
    print(f"Min total cost: {total_cost:.3f}")

    # Report chosen arcs and flows (suppress near-zeros)
    print("\nChosen arcs (y=1) and flows:")
    used_edges = []
    for u, v in G.edges:
        y = values.get(use[(u, v)])
        f = values.get(flow[(u, v)])
        if y > 0.5 or abs(f) > 1e-9:
            used_edges.append((u, v, f))
            print(
                f"  {u:>2} -> {v:<2} : y={int(round(y))}, f={f:.3f}, "
                f"cap={G[u][v]['capacity']}, vc={G[u][v]['vc']}, fc={G[u][v]['fc']}"
            )

    # --------------------------- Simple visualization -------------------------
    # Positions for plotting (spring layout keeps things readable)
    pos = nx.spring_layout(G, seed=7)

    # Node draw
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Edge styles: draw all edges thin, then overlay used ones thicker
    nx.draw_networkx_edges(
        G, pos, arrows=True, width=1.0, connectionstyle="arc3,rad=0.06"
    )

    # Overlay used edges with width proportional to flow
    if used_edges:
        widths = [
            max(1.5, 0.6 * f) for (_, _, f) in used_edges
        ]  # minimum width for visibility
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for (u, v, _) in used_edges],
            arrows=True,
            width=widths,
            connectionstyle="arc3,rad=0.06",
        )

        # Edge labels showing flow / capacity
        flow_labels = {
            (u, v): f"{values.get(flow[(u, v)]):.1f}/{G[u][v]['capacity']}"
            for (u, v, _) in used_edges
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=flow_labels, font_size=9)

    plt.title("Fixed-charge flow: thicker edges = more flow\n(label: flow/capacity)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def solve_set_cover_cp_sat():
    """
    Integral Set Cover:
      - Universe U of elements
      - Family S of subsets with integer costs cost[S]
    Vars:
      - z[S] ∈ {0,1} select subset S
    Constraints:
      - For each element u ∈ U: sum_{S: u∈S} z[S] >= 1
    Objective:
      - minimize sum_S cost[S]*z[S]
    """
    U = {1, 2, 3, 4, 5, 6}
    subsets = {
        "S1": {1, 2, 3},
        "S2": {2, 4},
        "S3": {3, 5, 6},
        "S4": {1, 4, 6},
        "S5": {2, 5},
        "S6": {4, 5, 6},
    }
    cost = {"S1": 4, "S2": 2, "S3": 3, "S4": 3, "S5": 2, "S6": 4}  # integers

    model = mo.Model(name="set_cover_cp_sat")

    # Decision vars
    z = {s: model.add_binary_variable(name=f"pick[{s}]") for s in subsets}

    # Cover constraints
    for u in U:
        model.add_linear_constraint(
            mo.fast_sum(z[s] for s in subsets if u in subsets[s]) >= 1,
            name=f"cover[{u}]",
        )

    # Objective
    model.set_objective(mo.fast_sum(cost[s] * z[s] for s in subsets), is_maximize=False)

    # Solve with CP-SAT
    params = mo.SolveParameters(time_limit=timedelta(seconds=10), enable_output=True)
    result = mo.solve(model, solver_type=mo.SolverType.CP_SAT, params=params)

    print(f"[SetCover] Termination: {result.termination.reason.name}")
    if not result.has_primal_feasible_solution():
        print("[SetCover] No feasible solution.")
        return

    vals = result.variable_values()
    chosen = [s for s in subsets if int(round(vals.get(z[s], 0.0)))]
    total_cost = sum(cost[s] for s in chosen)
    print(f"[SetCover] Min total cost: {total_cost}")
    print("[SetCover] Chosen subsets:", ", ".join(chosen))


if __name__ == "__main__":
    test_mathopt()
    test_mathopt_stigler_diet()
    # solve_fixed_charge_flow()
    solve_set_cover_cp_sat()
