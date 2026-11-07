"""Check that the automaton snippet accepts exactly the words the text traces."""

import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model


@pytest.mark.parametrize(
    "word, accepted",
    [([0, 1, 2, 0], True), ([1, 0, 1, 2], False), ([0, 0, 1, 1], False)],
)
def test_automaton_accepts(word, accepted):
    """0 -0-> 1 -1-> 2 -2-> 3 -0-> 3 ends in the final state 3; the others get
    stuck (no transition from state 0 on 1) or end in a non-final state."""
    namespace = run_snippet("advanced_modelling/automaton")
    model = namespace["model"]
    for variable, value in zip(namespace["transition_variables"], word, strict=True):
        model.add(variable == value)
    status = cp_model.CpSolver().solve(model)
    assert status == (cp_model.OPTIMAL if accepted else cp_model.INFEASIBLE)
