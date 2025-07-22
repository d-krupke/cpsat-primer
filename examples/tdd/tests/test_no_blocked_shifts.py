from nurserostering.nurse_vars import NurseDecisionVars
from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible
from _generate import create_shifts, create_nurse
from nurserostering.data_schema import NurseRosteringInstance
from nurserostering.modules import NoBlockedShiftsModule


def test_no_blocked_shifts_trivial():
    """
    Just create the variables for a nurse without any blocked shifts.
    This should be feasible and not raise any exceptions.
    """
    shifts = create_shifts(2)  # two consecutive shifts
    nurse = create_nurse("Nurse A", blocked_shifts=set())
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)
    with AssertModelFeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])


def test_no_blocked_shifts_infeasible():
    """
    Test that the model is infeasible when a nurse is assigned to a blocked shift.
    """
    shifts = create_shifts(2)  # two consecutive shifts
    nurse = create_nurse("Nurse A", blocked_shifts={shifts[0].uid})  # Blocked shift 0
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)
    with AssertModelInfeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])
        nurse_vars.fix(shifts[0].uid, True)


def test_no_blocked_shifts_feasible():
    """
    Test that the model is feasible when the assigned shift is not blocked.
    """
    shifts = create_shifts(2)  # two consecutive shifts
    nurse = create_nurse("Nurse A", blocked_shifts={shifts[0].uid})  # Blocked shift 1
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)
    with AssertModelFeasible() as model:
        nurse_vars = NurseDecisionVars(nurse, shifts, model)
        NoBlockedShiftsModule().build(instance, model, [nurse_vars])
        nurse_vars.fix(shifts[1].uid, True)  # This time we fix the other shift
