from nurserostering.modules import MinTimeBetweenShifts
from datetime import timedelta

from nurserostering.nurse_vars import NurseDecisionVars
from _generate import create_shifts, create_nurse
from nurserostering.data_schema import NurseRosteringInstance


from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible


def run_min_rest_test(
    assignments: list[bool | None],
    expected_feasible: bool,
    shift_length: int = 8,
    min_time_in_between: timedelta = timedelta(hours=16),
):
    shifts = create_shifts(len(assignments), shift_length=shift_length)
    nurse = create_nurse("Nurse A", min_time_between_shifts=min_time_in_between)
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    context = AssertModelFeasible() if expected_feasible else AssertModelInfeasible()
    with context as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        MinTimeBetweenShifts().build(instance, model, [nurse_vars])
        for s, assign in zip(shifts, assignments):
            if assign is None:
                continue  # skip free assignments
            nurse_vars.fix(s.uid, assign)


def test_pattern_false_true_true_false():
    run_min_rest_test(assignments=[None, True, True, None], expected_feasible=False)


def test_pattern_true_false_true_false():
    run_min_rest_test(
        assignments=[True, False, True, False],
        expected_feasible=True,
        shift_length=8,
        min_time_in_between=timedelta(hours=8),
    )


def test_pattern_true_false_false_true():
    run_min_rest_test(assignments=[True, False, False, True], expected_feasible=True)


def test_pattern_true_false_false_false_true():
    run_min_rest_test(
        assignments=[True, False, False, False, True], expected_feasible=True
    )


def test_pattern_all_false():
    run_min_rest_test(assignments=[False, False, False, False], expected_feasible=True)


def test_pattern_all_true_should_fail():
    run_min_rest_test(assignments=[True, True, True, True], expected_feasible=False)


def test_pattern_single_shift():
    run_min_rest_test(assignments=[True], expected_feasible=True)


def test_pattern_two_shifts_pause():
    run_min_rest_test(
        assignments=[True, False, True, False],
        expected_feasible=False,
        shift_length=8,
        min_time_in_between=timedelta(hours=16),
    )


def test_pattern_two_shifts_pause2():
    run_min_rest_test(
        assignments=[True, False, False, True],
        expected_feasible=True,
        shift_length=8,
        min_time_in_between=timedelta(hours=16),
    )
