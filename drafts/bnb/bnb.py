"""
A simple implementation of the branch and bound algorithm for the knapsack problem.
It is primarily intended to illustrate the algorithm and not for efficiency.
It is modular such that the performance difference between different strategies can be
explored. All strategies will be exponential in the worst case, but some strategies
will still result in significantly smaller branch and bound trees than others.
"""


import typing
import queue
from dataclasses import dataclass
import math

@dataclass
class Item:
    weight: int
    value: int


@dataclass
class Instance:
    items: typing.List[Item]
    capacity: int


BranchingDecisions = typing.List[typing.Optional[int]]


class FractionalSolution:
    def __init__(self, instance: Instance, selection: typing.List[float]):
        if not len(selection) == len(instance.items):
            raise ValueError("Selection must have same length as items.")
        self.instance = instance
        self.selection = selection

    def value(self) -> float:
        return sum(
            item.value * taken
            for item, taken in zip(self.instance.items, self.selection)
        )

    def weight(self) -> float:
        return sum(
            item.weight * taken
            for item, taken in zip(self.instance.items, self.selection)
        )

    def is_feasible(self) -> bool:
        return self.weight() <= self.instance.capacity

    def is_integral(self) -> bool:
        return all(taken == int(taken) for taken in self.selection)

    def __str__(self) -> str:
        return (
            "["
            + "|".join(f"{round(x,1) if x!=int(x) else int(x)}" for x in self.selection)
            + "]"
        )


class RelaxationSolver:
    def _infer(
        self, instance: Instance, fixation: BranchingDecisions
    ) -> BranchingDecisions:
        """
        Deduce further fixations from the given fixations.
        """
        fixation = fixation.copy()
        remaining_capacity = instance.capacity - sum(
            item.weight for item, x in zip(instance.items, fixation) if x == 1
        )
        for i, x in enumerate(fixation):
            if x is not None:
                continue
            # automatically fix all items to zero that are too heavy
            if instance.items[i].weight > remaining_capacity:
                fixation[i] = 0
        return fixation

    def solve(
        self, instance: Instance, fixation: BranchingDecisions
    ) -> FractionalSolution:
        fixation = self._infer(instance, fixation)  # deduce further fixations
        remaining_capacity = instance.capacity - sum(
            item.weight for item, x in zip(instance.items, fixation) if x == 1
        )
        # Compute solution
        selection = [1.0 if x == 1 else 0.0 for x in fixation]
        remaining_indices = [i for i, x in enumerate(fixation) if x is None]
        remaining_indices.sort(
            key=lambda i: instance.items[i].value / instance.items[i].weight,
            reverse=True,
        )
        for i in remaining_indices:
            # fill solution with items sorted by value/weight
            if instance.items[i].weight <= remaining_capacity:
                selection[i] = 1.0
                remaining_capacity -= instance.items[i].weight
            else:
                selection[i] = remaining_capacity / instance.items[i].weight
                break  # no capacity left
        assert all(
            x0 == x1 for x0, x1 in zip(fixation, selection) if x0 is not None
        ), "Fixed part is not allowed to change."
        return FractionalSolution(instance, selection)


class BnBNode:
    def __init__(
        self,
        relaxed_solution: FractionalSolution,
        branching_decisions: BranchingDecisions,
        depth: int,
        node_id: int,
    ) -> None:
        self.relaxed_solution = relaxed_solution
        self.branching_decisions = branching_decisions
        self.depth = depth
        self.node_id = node_id

    def __lt__(self, other: "BnBNode") -> bool:
        return self.node_id < other.node_id

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, BnBNode) and self.node_id == __value.node_id


class SearchStrategy:
    def __init__(self, priority: typing.Callable[[BnBNode], typing.Any]) -> None:
        self.queue = queue.PriorityQueue()
        self._priority = priority

    def add(self, node: BnBNode) -> None:
        self.queue.put((self._priority(node), node))

    def next(self) -> BnBNode:
        if self.has_next():
            return self.queue.get()[1]
        raise ValueError("No more nodes to explore.")

    def __len__(self) -> int:
        return self.queue.qsize()

    def nodes_in_queue(self) -> typing.Iterable[BnBNode]:
        return (node for _, node in self.queue.queue)

    def has_next(self) -> bool:
        return not self.queue.empty()


class NodeFactory:
    def __init__(self, instance: Instance, relaxation: RelaxationSolver) -> None:
        self._node_id_counter = 0
        self.instance = instance
        self.relaxation = relaxation

    def create_root(self) -> BnBNode:
        root = BnBNode(
            self.relaxation.solve(self.instance, [None] * len(self.instance.items)),
            [None] * len(self.instance.items),
            0,
            self._node_id_counter,
        )
        self._node_id_counter += 1
        return root

    def create_child(
        self, parent: BnBNode, branching_decisions: BranchingDecisions
    ) -> BnBNode:
        child = BnBNode(
            self.relaxation.solve(self.instance, branching_decisions),
            branching_decisions,
            parent.depth + 1,
            self._node_id_counter,
        )
        self._node_id_counter += 1
        return child

    def num_nodes(self) -> int:
        return self._node_id_counter


class BranchingStrategy:
    def make_decisions(self, node: BnBNode) -> typing.Iterable[BranchingDecisions]:
        frac_i = min(
            i for i, x in enumerate(node.relaxed_solution.selection) if x != int(x)
        )
        for value in [0, 1]:
            decisions = node.branching_decisions.copy()
            decisions[frac_i] = value
            yield decisions

class SolutionSet:
    def __init__(self) -> None:
        self._best_solution = None
        self._solutions = []

    def add(self, solution: FractionalSolution) -> None:
        assert solution.is_feasible() and solution.is_integral()
        if solution not in self._solutions:
            self._solutions.append(solution)
        if not self._best_solution or solution.value() > self._best_solution.value():
            self._best_solution = solution
        
    def best_solution_value(self) -> float:
        return self._best_solution.value() if self._best_solution else float("-inf")

    def best_solution(self) -> typing.Optional[FractionalSolution]:
        return self._best_solution

class Statistics:
    def __init__(
        self, node_factory: NodeFactory, search_strategy: SearchStrategy, solutions: SolutionSet
    ) -> None:
        self.node_factory = node_factory
        self.search_strategy = search_strategy
        self.solutions = solutions
        self._last_node = None

    def upper_bound(self) -> float:
        if not self.search_strategy.has_next():
            return self.solutions.best_solution_value() if self.solutions.best_solution() else float("inf")
        return max(
            self.search_strategy.nodes_in_queue(),
            key=lambda node: node.relaxed_solution.value(),
        ).relaxed_solution.value()

    def lower_bound(self) -> float:
        return self.solutions.best_solution_value()

    def report_node(self, node: BnBNode, status: str) -> None:
        self._last_node = (node, status)

    def report_new_solution(self, solution: FractionalSolution) -> None:
        self.best_solution = solution

    def print_header(self):
        print("Nodes\tDepth\tStatus\t\tValue\tUB\tLB")

    def print_progress(self):
        num_nodes = self.node_factory.num_nodes()
        num_nodes_in_queue = len(self.search_strategy)
        num_nodes_explored = num_nodes - num_nodes_in_queue
        last_node_value = (
            round(self._last_node[0].relaxed_solution.value())
            if self._last_node
            else "-"
        )
        last_node_depth = self._last_node[0].depth if self._last_node else "-"
        last_node_status = self._last_node[1] if self._last_node else "-\t"
        upper_bound = round(self.upper_bound(), 1)
        lower_bound = round(self.lower_bound(), 1)
        print(
            f"{num_nodes_explored}/{num_nodes}\t{last_node_depth}\t{last_node_status}\t{last_node_value}\t{upper_bound}\t{lower_bound}"
        )



class Heuristics:
    """
    Try to find solutions from fractional nodes using heuristics.
    """
    def __init__(self) -> None:
        pass

    def search(self, instance: Instance, node: BnBNode) -> typing.Iterable[FractionalSolution]:
        # Rounding down the fractional solution should yield a feasible solution.
        selection = [float(math.floor(x)) for x in node.relaxed_solution.selection]
        solution = FractionalSolution(instance, selection)
        if solution.is_feasible():
            yield solution

class BnBSearch:
    def __init__(
        self,
        instance: Instance,
        relaxation: RelaxationSolver,
        search_strategy: SearchStrategy,
        branching_strategy: BranchingStrategy,
    ) -> None:
        self.instance = instance
        self.solutions = SolutionSet()
        self.relaxation = relaxation
        self.node_factory = NodeFactory(instance, relaxation)
        self.search_strategy = search_strategy
        self.branching_strategy = branching_strategy
        self.statistics = Statistics(self.node_factory, self.search_strategy, self.solutions)
        self.heuristics = Heuristics()

    def search(self) -> typing.Optional[FractionalSolution]:
        root = self.node_factory.create_root()
        self.search_strategy.add(root)
        self.statistics.print_header()
        while self.search_strategy.has_next():
            self.statistics.print_progress()
            node = self.search_strategy.next()
            if not node.relaxed_solution.is_feasible():
                self.statistics.report_node(node, "infeasible")
                continue  # infeasibility prune
            if node.relaxed_solution.value() <= self.solutions.best_solution_value():
                self.statistics.report_node(node, "pruned   ")
                continue  # suboptimality prune
            if node.relaxed_solution.is_integral():
                self.statistics.report_node(node, "integral")
                # update best solution
                self.solutions.add(node.relaxed_solution)
                continue
            # try to find solutions using heuristics
            for heur_sol in self.heuristics.search(self.instance, node):
                self.solutions.add(heur_sol)
            # branch on a non-integer variable
            self.statistics.report_node(node, "branched")
            for decisions in self.branching_strategy.make_decisions(node):
                child = self.node_factory.create_child(node, decisions)
                self.search_strategy.add(child)
        self.statistics.print_progress()
        return self.solutions.best_solution()
