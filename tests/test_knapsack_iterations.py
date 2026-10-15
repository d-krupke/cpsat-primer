"""Regressions for the solver class shown in Coding Patterns."""

from itertools import product
from pathlib import Path
import re

import pytest
from ortools.sat.python import cp_model


def documented_solver(weights, values, capacity):
    text = (
        Path(__file__).resolve().parents[1] / "chapters/coding_patterns.md"
    ).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    namespace = {"cp_model": cp_model}
    exec(next(b for b in blocks if "class KnapsackInstance" in b), namespace)
    exec(
        next(b for b in blocks if "class MultiObjectiveKnapsackSolver" in b), namespace
    )
    instance = namespace["KnapsackInstance"](
        weights=weights, values=values, capacity=capacity
    )
    config = namespace["KnapsackSolverConfig"](time_limit=3, opt_tol=0)
    result = namespace["MultiObjectiveKnapsackSolver"](instance, config)
    result.solver.parameters.num_workers = 1
    return result


def test_repeated_solves_replace_hints():
    model = documented_solver([1, 2, 3], [4, 5, 6], 3)
    for _ in range(4):
        solution = model.solve()
        assert solution.objective == solution.upper_bound == 9
        assert model.model.validate() == ""


@pytest.mark.parametrize("ratio", [1.0, 0.95])
def test_maximum_value_retention(ratio):
    weights, values, capacity = [2, 3], [8, 9], 3
    model = documented_solver(weights, values, capacity)
    assert model.solve().objective == 9
    model.fix_current_objective(ratio)
    model.set_minimize_weight_objective()
    solution = model.solve()
    oracle = min(
        sum(w * b for w, b in zip(weights, bits))
        for bits in product(range(2), repeat=2)
        if sum(w * b for w, b in zip(weights, bits)) <= capacity
        and sum(v * b for v, b in zip(values, bits)) >= ratio * 9
    )
    assert solution.objective == oracle == 3


def test_minimum_weight_retention():
    model = documented_solver([2, 3], [8, 9], 3)
    model.set_minimize_weight_objective()
    model.model.add(sum(model.x) >= 1)
    assert model.solve().objective == 2
    model.fix_current_objective(1.1)
    model.set_maximize_value_objective()
    assert model.solve().objective == 8
