"""Run the regression-test pattern of the coding-patterns chapter for real.

The chapter's `add_test_case`/`test_saved_test_cases` store test cases next to
`__file__`; pointing `__file__` at this file uses the cases in `test_data/`.
"""

from pathlib import Path

from _snippets import run_snippet


def regression_tests(test_file: Path) -> dict:
    return run_snippet("coding_patterns/regression_tests", __file__=str(test_file))


def test_saved_test_cases():
    """The solver still agrees with the solutions saved in `test_data/knapsack`."""
    test_file = Path(__file__)
    assert len(list((test_file.parent / "test_data" / "knapsack").iterdir())) == 3
    regression_tests(test_file)["test_saved_test_cases"]()


def test_added_test_case_is_checked(tmp_path):
    """A freshly added case is saved in the expected layout and then passes."""
    namespace = regression_tests(tmp_path / "test_knapsack.py")
    instance = namespace["KnapsackInstance"](
        weights=[23, 31, 29, 44, 53], values=[92, 57, 49, 68, 60], capacity=100
    )
    config = namespace["KnapsackSolverConfig"](time_limit=5.0)
    namespace["add_test_case"](instance, config)
    (case,) = (tmp_path / "test_data" / "knapsack").iterdir()
    assert {p.name for p in case.iterdir()} == {
        "instance.json",
        "config.json",
        "solution.json",
    }
    namespace["test_saved_test_cases"]()
