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
        return sum(item.value * taken for item, taken in zip(self.instance.items, self.selection))
    
    def weight(self) -> float:
        return sum(item.weight * taken for item, taken in zip(self.instance.items, self.selection))
    
    def is_feasible(self) -> bool:
        return self.weight() <= self.instance.capacity
    
    def is_integral(self) -> bool:
        return all(taken == int(taken) for taken in self.selection)


def relaxation(
    instance: Instance, fixation: BranchingDecisions
) -> FractionalSolution:
    """
    Calculate the relaxed solution of the knapsack problem.
    Args:
    instance: The knapsack instance.
    fixation: Fixed part of the solution. None means not fixed, 0 means not taken, 1 means fully taken.
    Returns:
    A list of floats representing the relaxed solution of the knapsack problem and the bound of given fixation.

    """
    remaining_capacity = instance.capacity - sum(
        item.weight for item, x in zip(instance.items, fixation) if x == 1
    )
    assert remaining_capacity >= 0, "Fixation must not lead to exceeding capacity."
    # automatically fix all items to zero that are too heavy
    is_feasible = lambda i: fixation[i] == 1 or instance.items[i].weight <= remaining_capacity
    fixation = [x if is_feasible(i) else 0 for i, x in enumerate(fixation)]
    # Compute solution
    selection = [1.0 if x == 1 else 0.0 for x in fixation]
    remaining_indices = [i for i, x in enumerate(fixation) if x is None]
    remaining_indices.sort(key=lambda i: instance.items[i].value / instance.items[i].weight, reverse=True)
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
    def __init__(self, 
                 relaxed_solution: FractionalSolution, 
                 branching_decisions: BranchingDecisions,
                 depth: int,
                 node_id: int,
                 ) -> None:
        self.relaxed_solution = relaxed_solution
        self.branching_decisions = branching_decisions
        self.depth = depth
        self.node_id = node_id

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
    def __init__(self,
                 instance: Instance,
                relaxation: typing.Callable[[Instance, typing.List[typing.Optional[int]]], FractionalSolution]) -> None:
        self._node_id_counter = 0
        self.instance = instance
        self.relaxation = relaxation

    def create_root(self) -> BnBNode:
        root =  BnBNode(
            self.relaxation(self.instance, [None] * len(self.instance.items)),
            [None] * len(self.instance.items),
            0,
            self._node_id_counter,
        )
        self._node_id_counter += 1
        return root
    
    def create_child(self, parent: BnBNode, branching_decisions: BranchingDecisions) -> BnBNode:
        child = BnBNode(
            self.relaxation(self.instance, branching_decisions),
            branching_decisions,
            parent.depth + 1,
            self._node_id_counter,
        )
        self._node_id_counter += 1
        return child

    
class BranchingStrategy:

    def make_decisions(self, node: BnBNode) -> typing.Iterable[BranchingDecisions]:
        frac_i = min(i for i, x in enumerate(node.relaxed_solution.selection) if x != int(x))
        for value in [0, 1]:
            decisions = node.branching_decisions.copy()
            decisions[frac_i] = value
            yield decisions

class BnBSearch:
    def __init__(self,
                  instance: Instance,
                  relaxation: typing.Callable[[Instance, typing.List[typing.Optional[int]]], FractionalSolution],
                  search_strategy: SearchStrategy,
                  branching_strategy: BranchingStrategy,
                  ) -> None:
        self.instance = instance
        self.best_solution = FractionalSolution(instance, [0.0] * len(instance.items))
        self.relaxation = relaxation
        self.node_factory = NodeFactory(instance, relaxation)
        self.search_strategy = search_strategy
        self.branching_strategy = branching_strategy


    def lower_bound(self) -> float:
        return min(self.search_strategy.nodes_in_queue(),
                  key=lambda node: node.relaxed_solution.value()).relaxed_solution.value()
    
    def upper_bound(self) -> float:
        return self.best_solution.value()


    def search(self) -> FractionalSolution:
        root = self.node_factory.create_root()
        self.search_strategy.add(root)
        while self.search_strategy.has_next():
            node = self.search_strategy.next()
            if not node.relaxed_solution.is_feasible():
                continue  # infeasibility prune
            if node.relaxed_solution.value() <= self.best_solution.value():
                continue  # suboptimality prune
            if node.relaxed_solution.is_integral():
                # update best solution
                self.best_solution = node.relaxed_solution
                continue
            # branch on a non-integer variable
            for decisions in self.branching_strategy.make_decisions(node):
                child = self.node_factory.create_child(node, decisions)
                self.search_strategy.add(child)
        return self.best_solution
        