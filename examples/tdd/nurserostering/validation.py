"""
This module contains validation functions for nurse rostering solutions.
It is solver and algorithm agnostic, meaning it can be used to validate solutions
from any solver that produces a `NurseRosteringSolution` object. Just by providing
such functions, you have done a huge step towards computing a solution as now you
have a clean specification of what a valid and good solution looks like.
"""

from collections import defaultdict
from .data_schema import NurseRosteringInstance, NurseRosteringSolution


def assert_consistent_uids(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that all UIDs in the solution are part of the instance.
    """
    nurse_uids = {n.uid for n in instance.nurses}
    shift_uids = {s.uid for s in instance.shifts}
    for shift_uid, nurse_list in solution.nurses_at_shifts.items():
        if shift_uid not in shift_uids:
            raise AssertionError(f"Shift {shift_uid} is not present in the instance.")
        for nurse_uid in nurse_list:
            if nurse_uid not in nurse_uids:
                raise AssertionError(
                    f"Nurse {nurse_uid} is not present in the instance."
                )


def assert_no_blocked_shifts(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that no nurse is assigned to a blocked shift in the solution.
    """
    for nurse in instance.nurses:
        for shift_uid in nurse.blocked_shifts:
            if (
                shift_uid in solution.nurses_at_shifts
                and nurse.uid in solution.nurses_at_shifts[shift_uid]
            ):
                raise AssertionError(
                    f"Nurse {nurse.uid} is assigned to blocked shift {shift_uid}."
                )


def assert_demand_satisfaction(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that each shift meets its nurse demand.
    """
    for shift in instance.shifts:
        assigned = solution.nurses_at_shifts.get(shift.uid, [])
        if len(assigned) < shift.demand:
            raise AssertionError(
                f"Shift {shift.uid} demand not met: {len(assigned)}/{shift.demand} assigned."
            )


def assert_min_time_between_shifts(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
):
    """
    Assert that nurses are not assigned to shifts too close together.
    """
    shifts_by_uid = {s.uid: s for s in instance.shifts}
    nurse_to_shifts = defaultdict(list)
    for shift_uid, nurse_uids in solution.nurses_at_shifts.items():
        for nurse_uid in nurse_uids:
            nurse_to_shifts[nurse_uid].append(shifts_by_uid[shift_uid])
    for nurse in instance.nurses:
        assigned = sorted(nurse_to_shifts[nurse.uid], key=lambda s: s.start_time)
        for a, b in zip(assigned, assigned[1:]):
            if b.start_time < a.end_time + nurse.min_time_between_shifts:
                raise AssertionError(
                    f"Nurse {nurse.uid} assigned to shifts {a.uid} and {b.uid} with insufficient rest."
                )


def objective_value(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution
) -> int:
    """
    Calculate the objective value of the solution based on the instance's preferences and staff assignments.
    This function is to be implemented with CP-SAT.
    """
    nurses_by_uid = {n.uid: n for n in instance.nurses}
    obj_val = 0
    for shift_uid, nurse_uids in solution.nurses_at_shifts.items():
        for nurse_uid in nurse_uids:
            nurse = nurses_by_uid[nurse_uid]
            if shift_uid in nurse.preferred_shifts:
                # This is a minimization problem, so we we subtract the weight for preferred shifts
                obj_val -= nurse.preferred_shift_weight
            if not nurse.staff:
                # Add the penalty for assigning a non-staff nurse (they are more expensive)
                obj_val += instance.staff_weight
    return obj_val


def assert_solution_is_feasible(
    instance: NurseRosteringInstance,
    solution: NurseRosteringSolution,
    check_objective: bool = True,
):
    """
    Run all standard feasibility checks.
    """
    assert_consistent_uids(instance, solution)
    assert_no_blocked_shifts(instance, solution)
    # assert_shift_limits(instance, solution)
    assert_demand_satisfaction(instance, solution)
    assert_min_time_between_shifts(instance, solution)
    if check_objective:
        obj_val = objective_value(instance, solution)
        if obj_val != solution.objective_value:
            raise AssertionError(
                f"Objective value mismatch: expected {obj_val}, got {solution.objective_value}."
            )
