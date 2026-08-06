import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import typer

    app = typer.Typer(help="CLI for LNS Knapsack example", no_args_is_help=True)
    return app, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LNS Example 1: Knapsack

    This notebook is an exploration of Large Neighbourhood Search (LNS). The first example is an application of the knapsack problem.

    In the Knapsack Problem we have a set of items $I$ with a weight $w_i$ and a value $v_i$ for each item $i \in I$. The goal is to select a subset of items such that the total weight does not exceed a given capacity $C$, while maximizing the total value.

    $$\max \sum_{i \in I} v_i x_i $$

    $$ \text{s.t.} \sum_{i \in I} \leq C $$

    $$ x_i \in \{ 0, 1 \}  $$

    This is one of the simplest NP-hard problem to formulate. It can be solved in pseudo-polynomial time using a dynamic programming approach. Also,
    CP-SAT is also able to generate instantaneous solutions to many large instances of this problem.

    Due to its simple structure it is an ideal example to demonstrate the use of Large Neighbourhood Search.
    """)
    return


@app.cell
def _():
    # import all dependencies
    import typing
    import math
    import random

    from ortools.sat.python import cp_model
    return cp_model, math, random, typing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Instance Generation

    First we need to create some classes that will generate random instances for the knapsack problem.
    """)
    return


@app.cell
def _(math, random, typing):
    class Item:
        """
        A simple class to represent an item in the knapsack problem.
        Every instance of this class is unique, i.e., two items with
        the same weight and value are not equal. Otherwise, we could
        only have a single item for each weight and value combination.
        """

        def __init__(self, weight: int, value: int):
            self.weight = weight
            self.value = value

        def __repr__(self):
            return f"Item({self.weight}, {self.value})"

        def __eq__(self, other):
            return id(self) == id(other)

        def __hash__(self):
            return id(self)

    class Instance:
        """
        Simple instance container.
        """

        def __init__(self, items: typing.List[Item], capacity: int) -> None:
            self.items = items
            self.capacity = capacity
            assert len(items) > 0
            assert capacity > 0


    def random_instance(num_items: int, ratio: float) -> Instance:
        """
        Creates a random instance of the knapsack problem.
        :param num_items: The number of items.
        :param ratio: The ratio between capacity and sum of weights.
        :return: A list of items and a capacity
        """
        items = []
        for i in range(num_items):
            weight = random.randint(10, 1000)
            value = round(random.triangular(1, 100, 5) * weight)
            items.append(Item(weight, value))
        capacity = math.ceil(sum(item.weight for item in items) * ratio)
        return Instance(items, capacity)

    def value(items: typing.List[Item]) -> int:
        """
        Returns the total value of a list of items.
        """
        return sum(item.value for item in items)
    return Instance, Item, random_instance, value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Greedy Algorithm

    Next, we need an initial solution. We use a simple greedy algorithm that adds items to the knapsack as long as the capacity is not exceeded.
    A better greedy implementation would be to sotrt the items by value/weight ratio and add the items with the highest ratio first. However, this ofter crease near-optimal solutions and we wouldn't see much improvement from the LNS.
    """)
    return


@app.cell
def _(Instance, Item, typing):
    def greedy_solution(instance: Instance) -> typing.List[Item]:
        """
        A simple greedy algorithm for the knapsack problem.
        It is bad on purpose, so we can improve it with local search.
        For random instances, the greedy algorithm otherwise often
        finds the (nearly) optimal solution and there is nothing to see.
        """
        solution = []
        remaining_capacity = instance.capacity
        for item in instance.items:
            if item.weight <= remaining_capacity:
                solution.append(item)
                remaining_capacity -= item.weight
        return solution
    return (greedy_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exact Solver for Subproblem

    We will remove itmes from the knapsack and try to refill it with better items. This subproblem is the Knapsack problem again, so we can reuse the CP-SAT solver to build solutions.
    """)
    return


@app.cell
def _(Instance, Item, cp_model, typing):
    def solve_knapsack(
        instance: Instance, max_time_in_seconds: float = 90, log_progress: bool = False
    ) -> typing.List[Item]:
        """
        Optimal solver for knapsack
        """
        model = cp_model.CpModel()

        # decision variables
        x = [model.new_bool_var(f"x_{i}") for i in range(len(instance.items))]

        # capacity constraint
        model.add(
            sum(x[i] * item.weight for i, item in enumerate(instance.items))
            <= instance.capacity
        )

        # objective function
        model.maximize(sum(x[i] * item.value for i, item in enumerate(instance.items)))

        # solver invokation
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        solver.parameters.log_search_progress = log_progress
        status = solver.solve(model)

        # extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if status == cp_model.FEASIBLE:
                print(
                    "Warning: Solver did not find optimal solution. Returned solution is feasible but not optimal."
                )
            return [
                item for i, item in enumerate(instance.items) if solver.value(x[i]) == 1
            ]
        else:
            return []
    return (solve_knapsack,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LNS for Knapsack

    ## Initialization

    Start with an initial solution that is then improved by the LNS algorithm.
    We use the simple greedy heuristic from above to generate an initial solution.
    """)
    return


@app.cell
def _(greedy_solution, random_instance):
    # Create instance
    instance = random_instance(10_000, 0.1)

    # compute some initial solution
    initial_solution = greedy_solution(instance)
    return initial_solution, instance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Improve the solution with LNS

    The LNS algorithms is a heuristic that iteratively destroys and repairs parts of the solution. We remove a part of the selected item in the current solution nad then select some additional items fromr the remaining set. Using the exact solver, we find teh optimal solution for the remaining capacity using teh selected items. This is repeated for several iteratios.

    There are two important parameters for the LNS algorithm:
    * The size of the subproblem we solve with the exact solver.
    * The number of items we remove from the current solution.
    """)
    return


@app.cell
def _(Instance, Item, random, solve_knapsack, typing):
    class KnapsackLns:
        """
        Knapsack LNS solver.
        """

        def __init__(
            self,
            instance: Instance,
            initial_solution: typing.List[Item],
            subproblem_size: int,
        ):
            self.instance = instance
            self.solution = initial_solution
            self.subproblem_size = (
                subproblem_size  # Number of items to consider in subproblem
            )
            self.solutions = [initial_solution]

        def _remaining_capacity(self):
            """
            Instance capacity minus the weight of all items in the current solution
            """
            return self.instance.capacity - sum(item.weight for item in self.solution)

        def _remaining_items(self):
            """
            List of items that are not included in the current solution
            """
            return list(set(self.instance.items).difference(self.solution))

        def _destroy(self, num_items: int) -> typing.List[Item]:
            """
            Destroy a part of the solution by removing num_items from it.
            """
            num_items = min(len(self.solution), num_items)
            assert 0 <= num_items <= self.subproblem_size
            items_removed = random.sample(self.solution, num_items)
            self.solution = [
                item for item in self.solution if item not in items_removed
            ]
            print(
                f"Removed {len(items_removed)} items from solution. New remaining capacity: {self._remaining_capacity()}"
            )
            return items_removed

        def _repair(self, I_: typing.List[Item], max_time_in_seconds: float = 90):
            """
            Repair the solution by adding items from I_ to it.
            """
            C_ = self._remaining_capacity()
            print(
                f"Repairing solution by considering {len(I_)} items to fill the remaining capacity of {C_}."
            )
            subsolution = solve_knapsack(Instance(I_, C_), max_time_in_seconds)
            self.solution += subsolution
            assert self._remaining_capacity() >= 0

        def perform_lns_iteration(
            self, destruction_size: int, max_time_in_seconds: float = 90
        ):
            # 1. Destroy
            assert destruction_size > 0
            items_removed = self._destroy(destruction_size)
            # 2. Build subproblem for repair
            remaining_items = self._remaining_items()
            n = min(self.subproblem_size - destruction_size, len(remaining_items))
            new_items_to_consider = random.sample(remaining_items, n)
            # Add the removed items to the set of items to consider, such that
            # we can also find an equally good solution
            I_ = list(
                set(items_removed + new_items_to_consider).difference(self.solution)
            )
            # 3. Repair
            self._repair(I_, max_time_in_seconds)
            self.solutions.append(self.solution)
    return (KnapsackLns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the LNS

    Play around with the parameters and see how the LNS algorithm improves the solution
    """)
    return


@app.cell
def _(KnapsackLns, initial_solution, instance, mo, value):
    # don't run this cell if you're at the CLI, but run it if you're in the notebook
    if mo.app_meta().mode != "script":
        lns = KnapsackLns(instance, initial_solution, subproblem_size=1000)
        for i in range(25):
            lns.perform_lns_iteration(destruction_size=100)
            print(
                f"=> Iteration {i}: {value(lns.solution)} (improvement: {value(lns.solution) / value(lns.solutions[0])})"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use CP-SAT to find an optimal solution

    For comparison, we can apply the CP-SAT solver to the whole knapsack problem and see what the optimal solution is.
    """)
    return


@app.cell
def _(Instance, instance, solve_knapsack, value):
    def run_cpsat(knapsack_instance: Instance, max_solver_time: int = 90):
        """
        Run CP-SAT for Knapsack
        """
        optimal_solution = solve_knapsack(instance, max_time_in_seconds=max_solver_time)
        print(f"CP-SAT solution: {value(optimal_solution)}")
    return (run_cpsat,)


@app.cell
def _(instance, mo, run_cpsat):
    # don't run this cell if you're at the CLI, but run it if you're in the notebook
    if mo.app_meta().mode != "script":
        run_cpsat(knapsack_instance=instance, max_solver_time=90)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    We are able to improve the initial solution via LNS, however, only because we used a bad greedy algorithm.
    If we had used a better greedy algorithm, the LNS algorithm would not be able to improve the solution by much.
    However, the LNS algorithm is a very powerful heuristic that can be used to improve solutions for many problems.
    This example just had the purpose to demonstrate the implementation of LNS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise

    Try to generalize the LNS algorithm for Multi-Knapsack problems, where instead of a single knapsack, we have multiple knapsacks with different capacities, and items can have different values and weights for each knapsack.
    Multi-Knapsack problems can be significantly harder but also of practical interest for many applications, such as scheduling and resource allocation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Typer entry point

    We've defined two entry point functions here:
    * `lnsk` generates a random knapsack instance then runs the LNS and also compares it to a CP-SAT solution
    * `cpsatk` generates a random knapsack instance and runs just the CP-SAT solver


    To run these functions as a CLI tool, go to your terminal and try:
    ```
    > python file/path/lns_marimo.py --help
    ```
    """)
    return


@app.cell
def _(KnapsackLns, app, greedy_solution, random_instance, run_cpsat, value):
    @app.command()
    def lnsk(
        numitems: int,
        cwratio: float = 0.1,
        iterations: int = 25,
        destroy: int = 100,
        subproblem: int = 1000,
        cptimeout: int = 90,
    ):
        """
        Run LNS for Knapsack:
        choose NUMITEMS for random instance,
        and CWRATIO for capacity to weight ratio
        and ITERATIONS for LNS
        """

        # Create instance
        instance = random_instance(num_items=numitems, ratio=cwratio)

        # compute some initial solution
        initial_solution = greedy_solution(instance)

        lns = KnapsackLns(instance, initial_solution, subproblem_size=subproblem)
        for i in range(iterations):
            lns.perform_lns_iteration(destruction_size=destroy)
            print(
                f"=> Iteration {i+1}: {value(lns.solution)} (improvement: {value(lns.solution) / value(lns.solutions[0])})"
            )

        print(
            "\nHold my beer while I solve this with the CPSAT Solver for comparison.\n\n"
        )

        run_cpsat(knapsack_instance=instance, max_solver_time=cptimeout)

    @app.command()
    def cpsatk(numitems: int, cwratio: float=0.1, cptimeout: int = 90):
        """
        Run CPSAT for Knapsack:
        choose NUMITEMS for random instance,
        and CPRATIO for capacity to weight ratio
        and CPTIMEOUT for solver timeout
        """
        # Create instance
        instance = random_instance(num_items=numitems, ratio=cwratio)

        print(
            "\nRandom knapsack instance generated. Now solving with CP-SAT solver...\n\n"
        )

        run_cpsat(knapsack_instance=instance, max_solver_time=cptimeout)

    return


@app.cell
def _(app, mo):
    if mo.app_meta().mode == "script":
        app()
    return


if __name__ == "__main__":
    app.run()
