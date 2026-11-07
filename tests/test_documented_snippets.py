"""Exercise the snippets shown in the chapters, not independently maintained copies.

Snippets are marked with `<!-- snippet: ... -->` comments in `chapters/*.md`;
see `_snippets.py` for the marker format. Every marked snippet is smoke-tested
here, which catches API changes in new OR-Tools releases; the remaining tests
check that a snippet computes what the surrounding text claims.
"""

from itertools import product

import pytest
from _snippets import documented_snippets, run_snippet
from ortools.sat.python import cp_model


def solver():
    result = cp_model.CpSolver()
    result.parameters.num_workers = 1
    result.parameters.max_time_in_seconds = 3
    return result


@pytest.mark.parametrize("snippet_id", sorted(documented_snippets()))
def test_snippet_runs(snippet_id):
    """Every marked snippet runs (or, with `mode=compile`, compiles) as printed."""
    snippet = documented_snippets()[snippet_id]
    if snippet.mode == "compile":
        snippet.compile()
    else:
        run_snippet(snippet_id)


def test_introductory_example():
    assert run_snippet("example/first_model")["solver"].objective_value == 1500


def test_arithmetic_model_is_valid():
    """The divisor's domain excludes zero, as CP-SAT requires."""
    assert run_snippet("modelling/arithmetic")["model"].validate() == ""


@pytest.mark.parametrize("numerator, divisor", list(product([-5, 5], [-2, 2])))
def test_signed_division_and_modulo(numerator, divisor):
    """Division truncates toward zero and the remainder follows the numerator's
    sign, as the text explains (unlike Python's `//` and `%`)."""
    model = cp_model.CpModel()
    quotient = model.new_int_var(-10, 10, "quotient")
    remainder = model.new_int_var(-10, 10, "remainder")
    model.add_division_equality(quotient, numerator, divisor)
    model.add_modulo_equality(remainder, numerator, 3)
    s = solver()
    assert s.solve(model) == cp_model.OPTIMAL
    assert s.value(quotient) == (2 if numerator * divisor > 0 else -2)
    assert s.value(remainder) == (2 if numerator > 0 else -2)


@pytest.mark.parametrize("values", [(-3, -4, -5), (0, -100, 0), (100, 100, 100)])
def test_absolute_and_extrema_domains(values):
    """The auxiliary domains admit all extremes, including all-negative inputs."""
    namespace = run_snippet("modelling/abs_min_max")
    model = namespace["model"]
    for name, value in zip(["x", "y", "z"], values):
        model.add(namespace[name] == value)
    s = solver()
    assert s.solve(model) == cp_model.OPTIMAL
    x, y, z = values
    assert s.value(namespace["abs_xz"]) == abs(x + z)
    assert s.value(namespace["max_xyz"]) == max(x, y - 1, z)
    assert s.value(namespace["min_xyz"]) == min(x, y - 1, z)


@pytest.mark.parametrize("a, b", list(product(range(2), repeat=2)))
def test_boolean_product_penalty(a, b):
    """Minimizing the product cannot undercut b1*b2, thanks to the OR clause."""
    namespace = run_snippet("modelling/boolean_product")
    model = namespace["model"]
    model.add(namespace["b1"] == a)
    model.add(namespace["b2"] == b)
    model.minimize(namespace["b1b2"])
    s = solver()
    assert s.solve(model) == cp_model.OPTIMAL
    assert s.objective_value == a * b


def test_automaton_snippet():
    assert solver().solve(run_snippet("advanced_modelling/automaton")["model"]) == (
        cp_model.OPTIMAL
    )


def test_mathopt_expression_handles():
    namespace = run_snippet("mathopt/expression_handles")
    assert len(namespace["in_vars"]) == len(namespace["out_vars"]) == 5


def test_mathopt_set_cover_is_optimal():
    """The cheapest covers cost 8, e.g. S2 + S3 + S4 or S1 + S6."""
    namespace = run_snippet("mathopt/set_cover")
    assert namespace["total_cost"] == 8
