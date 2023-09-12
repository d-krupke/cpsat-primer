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
    """ 
    Represents an item with a weight and a value for the knapsack problem.
    """
    weight: int
    value: int


@dataclass
class Instance:
    """ 
    Represents an instance with a list of items and capacity of the knapsack problem.
    """
    items: typing.List[Item]
    capacity: int

# List of the item fixations. None means not fixed, 0 means not taken, 1 means fully taken.
BranchingDecisions = typing.List[typing.Optional[int]]


class FractionalSolution:
    """
    Represents a fractional solution to the knapsack problem.
    """

    def __init__(self, instance: Instance, selection: typing.List[float]):
        if not len(selection) == len(instance.items):
            raise ValueError("Selection must have same length as items.")
        self.instance = instance
        self.selection = selection

  
    def value(self) -> float:
        """
        Total value of fractional solution.
        """
        return sum(
            item.value * taken
            for item, taken in zip(self.instance.items, self.selection)
        )
    
    def weight(self) -> float:
        """
        Total weight of fractional solution.
        """
        return sum(
            item.weight * taken
            for item, taken in zip(self.instance.items, self.selection)
        )

    def is_feasible(self) -> bool:
        """
        Check if total weight of fractional solution doesn't exceed knapsack capacity.
        """
        return self.weight() <= self.instance.capacity
    
    def is_integral(self) -> bool:
        """
        Check if all item selections of fractional solution are integers.
        """
        return all(taken == int(taken) for taken in self.selection)

    def __str__(self) -> str:
        return (
            "["
            + "|".join(f"{round(x,1) if x!=int(x) else int(x)}" for x in self.selection)
            + "]"
        )


class RelaxationSolver:
    """
    Solve the fractional knapsack problem from the given instance and branching decisions.
    """
    def _infer(
        self, instance: Instance, fixation: BranchingDecisions
    ) -> BranchingDecisions:
        """
        Deduce further fixations from the given fixations and instance.
        """
        fixation = fixation.copy()
        remaining_capacity = instance.capacity - sum(
            item.weight for item, x in zip(instance.items, fixation) if x == 1
        )
        for i, x in enumerate(fixation):
            if x is not None:
                continue
            # Automatically fix all items to zero that are too heavy
            if instance.items[i].weight > remaining_capacity:
                fixation[i] = 0
        return fixation

    def solve(
        self, instance: Instance, fixation: BranchingDecisions
    ) -> FractionalSolution:
        """
        Solve the fractional knapsack problem from the given instance and deduced fixations.
        """
        fixation = self._infer(instance, fixation)  # Deduce further fixations
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
            # Fill solution with items sorted by value/weight
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
    """
    Represent a node relaxed_soluattributes  with in the branch-and-bound search tree.
    """
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
        """
        Compare two nodes based on their node IDs.
        """
        return self.node_id < other.node_id
    
    def __eq__(self, __value: object) -> bool:
        """
        Check if this node is equal to another object.
        """
        return isinstance(__value, BnBNode) and self.node_id == __value.node_id


class SearchStrategy:
    """
    Manage the nodes of branch-and-bound search tree with priority queue.
    """
    def __init__(self, priority: typing.Callable[[BnBNode], typing.Any]) -> None:
        self.queue = queue.PriorityQueue()
        self._priority = priority

    def add(self, node: BnBNode) -> None:
        """
        Add a node to the priority queue.
        """
        self.queue.put((self._priority(node), node))

    def next(self) -> BnBNode:
        """
        Get the next node from the priority queue.
        """
        if self.has_next():
            return self.queue.get()[1]
        raise ValueError("No more nodes to explore.")

    def __len__(self) -> int:
        """
        Get the number of nodes in the priority queue.
        """
        return self.queue.qsize()

    def nodes_in_queue(self) -> typing.Iterable[BnBNode]:
        """
        Get a iterable of nodes in the priority queue.
        """
        return (node for _, node in self.queue.queue)
    
    def has_next(self) -> bool:
        """
        Check if there are more nodes to explore in the priority queue.
        """
        return not self.queue.empty()


class NodeFactory:
    """
    Create nodes for the search tree.
    """
    def __init__(self, instance: Instance, relaxation: RelaxationSolver) -> None:
        self._node_id_counter = 0
        self.instance = instance
        self.relaxation = relaxation

    def create_root(self) -> BnBNode:
        """
        Create and return the root node of the search tree
        """
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
        """
        Create a child node based on the parent node and branching decisions.
        """
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
    """
    Create decision branches based on the fractional solution of the given node.
    """
    def make_decisions(self, node: BnBNode) -> typing.Iterable[BranchingDecisions]:
        """
        Branch on a non-integer variable.
        """
        frac_i = min(
            i for i, x in enumerate(node.relaxed_solution.selection) if x != int(x)
        )
        for value in [0, 1]:
            decisions = node.branching_decisions.copy()
            decisions[frac_i] = value
            yield decisions

class SolutionSet:
    """
    Store feasible found solutions,
    determine and keep track the best solution among them. 
    """
    def __init__(self) -> None:
        self._best_solution = None
        self._solutions = []

    def add(self, solution: FractionalSolution) -> None:
        """
        Add a feasible and integral solution to the solution set and
        update the best solution if necessary.
        """
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
    """
    Track and report various statistical information related to the branch-and-bound search.
    """
    def __init__(
        self, node_factory: NodeFactory, search_strategy: SearchStrategy, solutions: SolutionSet
    ) -> None:
        self.node_factory = node_factory
        self.search_strategy = search_strategy
        self.solutions = solutions
        self._last_node = None

    def upper_bound(self) -> float:
        """
        Get maximum solution value of nodes in priority queue, if the queue isn't empty;
        otherwise, the best solution value in the solution set.
        """
        if not self.search_strategy.has_next():
            return self.solutions.best_solution_value() if self.solutions.best_solution() else float("inf")
        return max(
            self.search_strategy.nodes_in_queue(),
            key=lambda node: node.relaxed_solution.value(),
        ).relaxed_solution.value()

    def lower_bound(self) -> float:
        """
        Get the best solution value in the solution set, 
        """
        return self.solutions.best_solution_value()

    def report_node(self, node: BnBNode, status: str) -> None:
        """
        Report the status of the given node.
        """
        self._last_node = (node, status)

    # I don't know if this function is necessary because it may never be used ???
    def report_new_solution(self, solution: FractionalSolution) -> None:
        self.best_solution = solution

    def print_header(self):
        print("Nodes\tDepth\tStatus\t\tValue\tUB\tLB")

    def print_progress(self):
        """
        Print the progress information of the search process, which
        includes the explored nodes number,the created nodes number,
        the depth and status of the current node,
        the upper bound, and the lower bound.
        """
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
    Try to find feasible solution from given nodes using heuristics.
    """
    def __init__(self) -> None:
        pass

    def search(self, instance: Instance, node: BnBNode) -> typing.Iterable[FractionalSolution]:
        """
        Yield a feasible solution by rounding down the fractional solution of the given node.
        """
        selection = [float(math.floor(x)) for x in node.relaxed_solution.selection]
        solution = FractionalSolution(instance, selection)
        if solution.is_feasible():
            yield solution

class BnBSearch:
    """
    Perform the branch-and-bound search to determine the fractional solution for
    a specified instance of the knapsack problem.
    """
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
        """
        Perform a branch-and-bound search to find the optimal fractional solution
        for the knapsack problem instance.
        """

        # the branch-and-bound search start from the root node and
        # continue until the search strategy has no more nodes to explore.
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
    
