"""Exercise the canonical snippets, not independently maintained copies."""

from itertools import product
from pathlib import Path
import re

from ortools.sat.python import cp_model


CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"


def python_blocks(chapter):
    return re.findall(
        r"```python\n(.*?)```",
        (CHAPTERS / f"{chapter}.md").read_text(encoding="utf-8"),
        re.S,
    )


def solver():
    result = cp_model.CpSolver()
    result.parameters.num_workers = 1
    result.parameters.max_time_in_seconds = 3
    return result


def test_introductory_example():
    namespace = {}
    exec(python_blocks("example")[0], namespace)
    assert namespace["solver"].objective_value == 1500


def test_signed_arithmetic():
    code = next(b for b in python_blocks("modelling") if "xyz =" in b)
    for numerator, divisor in product([-5, 5], [-2, 2]):
        namespace = {"cp_model": cp_model}
        exec(code, namespace)
        model = namespace["model"]
        assert model.validate() == ""
        # Test each documented operation separately: their targets need not coincide.
        model = cp_model.CpModel()
        quotient = model.new_int_var(-10, 10, "quotient")
        remainder = model.new_int_var(-10, 10, "remainder")
        model.add_division_equality(quotient, numerator, divisor)
        model.add_modulo_equality(remainder, numerator, 3)
        s = solver()
        assert s.solve(model) == cp_model.OPTIMAL
        assert s.value(quotient) == (2 if numerator * divisor > 0 else -2)
        assert s.value(remainder) == (2 if numerator > 0 else -2)


def test_absolute_and_extrema_domains():
    code = next(b for b in python_blocks("modelling") if "abs_xz =" in b)
    for values in [(-3, -4, -5), (0, -100, 0), (100, 100, 100)]:
        namespace = {"cp_model": cp_model}
        exec(code, namespace)
        model = namespace["model"]
        for name, value in zip(["x", "y", "z"], values):
            model.add(namespace[name] == value)
        s = solver()
        assert s.solve(model) == cp_model.OPTIMAL
        x, y, z = values
        assert s.value(namespace["abs_xz"]) == abs(x + z)
        assert s.value(namespace["max_xyz"]) == max(x, y - 1, z)
        assert s.value(namespace["min_xyz"]) == min(x, y - 1, z)


def test_boolean_product_penalty():
    code = next(b for b in python_blocks("modelling") if '"b1b2"' in b)
    for a, b in product(range(2), repeat=2):
        namespace = {"cp_model": cp_model}
        exec(code, namespace)
        model = namespace["model"]
        model.add(namespace["b1"] == a)
        model.add(namespace["b2"] == b)
        model.minimize(namespace["b1b2"])
        s = solver()
        assert s.solve(model) == cp_model.OPTIMAL
        assert s.objective_value == a * b


def test_automaton_snippet():
    code = next(
        b for b in python_blocks("advanced_modelling") if "transition_triples=" in b
    )
    namespace = {"cp_model": cp_model}
    exec(code, namespace)
    s = solver()
    assert s.solve(namespace["model"]) == cp_model.OPTIMAL


def test_mathopt_expression_handles():
    from ortools.math_opt.python import mathopt as mo

    namespace = {"mo": mo, "model": mo.Model()}
    code = next(b for b in python_blocks("mathopt") if "incoming_flow =" in b)
    exec(code, namespace)
    assert len(namespace["in_vars"]) == len(namespace["out_vars"]) == 5


def test_mathopt_complete_examples_compile():
    for code in python_blocks("mathopt")[-2:]:
        compile(code, "mathopt example", "exec")
