from collections import defaultdict
from nurserostering.data_schema import NurseRosteringInstance, NurseRosteringSolution


def print_instance_and_solution(
    instance: NurseRosteringInstance, solution: NurseRosteringSolution | None = None
):
    print("\n=== Nurse Rostering Instance ===")
    print(f"- Nurses: {len(instance.nurses)}")
    print(f"- Shifts: {len(instance.shifts)}")
    print(f"- Staff weight: {instance.staff_weight}")
    print("\nNurses:")
    for nurse in instance.nurses:
        status = "Staff" if nurse.staff else "Contractor"
        print(f"  • {nurse.name} [{status}, rest ≥ {nurse.min_time_between_shifts}]")
        if nurse.preferred_shifts:
            print(
                f"     ↳ Preferences: {len(nurse.preferred_shifts)} shift(s), weight {nurse.preferred_shift_weight}"
            )
        if nurse.blocked_shifts:
            print(f"     ↳ Blocked: {len(nurse.blocked_shifts)} shift(s)")

    print("\nShifts:")
    for shift in instance.shifts:
        print(
            f"  • {shift.name}: {shift.start_time.strftime('%a %Y-%m-%d %H:%M')} — {shift.end_time.strftime('%H:%M')} (Demand: {shift.demand}"
        )

    if solution is None:
        return

    print("\n=== Solution ===")
    print(f"- Objective value: {solution.objective_value}")
    print(f"- Timestamp: {solution.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAssignments:")

    # Invert nurses_at_shifts to print per nurse
    shifts_by_uid = {s.uid: s for s in instance.shifts}
    assignments_by_nurse = defaultdict(list)
    for shift_uid, nurse_uids in solution.nurses_at_shifts.items():
        for nurse_uid in nurse_uids:
            assignments_by_nurse[nurse_uid].append(shift_uid)

    for nurse in instance.nurses:
        assigned_shift_uids = sorted(
            assignments_by_nurse[nurse.uid],
            key=lambda uid: shifts_by_uid[uid].start_time,
        )
        print(f"  • {nurse.name}: {len(assigned_shift_uids)} shift(s)")
        for uid in assigned_shift_uids:
            shift = shifts_by_uid[uid]
            print(f"     ↳ {shift.name} ({shift.start_time.strftime('%a %H:%M')})")

    print()
    print("Shifts and assigned nurses:")
    for shift in instance.shifts:
        assigned_nurses = [
            nurse.name
            for nurse_uid in solution.nurses_at_shifts.get(shift.uid, [])
            for nurse in instance.nurses
            if nurse.uid == nurse_uid
        ]
        print(
            f"  • {shift.name}: {', '.join(assigned_nurses) if assigned_nurses else 'No nurses assigned'}"
        )
