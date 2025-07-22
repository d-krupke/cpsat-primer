from nurserostering.output import print_instance_and_solution
from nurserostering.generate_random_instance import generate_random_instance
from nurserostering.solver import NurseRosteringModel
from nurserostering.validation import (
    assert_solution_is_feasible,
)
from datetime import datetime, timedelta
from nurserostering.data_schema import Nurse, Shift, NurseRosteringInstance


def test_random_instances():
    """
    Test the generation of random nurse rostering instances.
    This function creates a random instance and checks if it can be solved.
    """

    instance = generate_random_instance(num_nurses=12, num_days=14, staff_ratio=0.5)

    model = NurseRosteringModel(instance)
    solution = model.solve()
    assert_solution_is_feasible(instance, solution)
    assert solution is not None, "The solution should not be None"
    assert len(solution.nurses_at_shifts) > 0, (
        "There should be at least one nurse assigned to a shift"
    )


def test_fixed_instance():
    """
    === Nurse Rostering Instance ===
    - Nurses: 8
    - Shifts: 21 over 1 period(s)
    - Staff weight: 2

    Nurses:
    • Alice (prefers mornings, no Sundays) [Staff, max 6/period, rest ≥ 10:00:00]
        ↳ Preferences: 7 shift(s), weight 3
        ↳ Blocked: 3 shift(s)
    • Bob (prefers nights) [Staff, max 6/period, rest ≥ 10:00:00]
        ↳ Preferences: 7 shift(s), weight 3
    • Clara (prefers weekends, blocks weekdays) [Staff, max 4/period, rest ≥ 10:00:00]
        ↳ Preferences: 6 shift(s), weight 2
        ↳ Blocked: 15 shift(s)
    • Dan (no preferences) [Staff, max 6/period, rest ≥ 8:00:00]
    • Eve (contractor, prefers day shifts) [Contractor, max 5/period, rest ≥ 10:00:00]
        ↳ Preferences: 7 shift(s), weight 2
    • Frank (prefers day shifts) [Staff, max 6/period, rest ≥ 10:00:00]
        ↳ Preferences: 7 shift(s), weight 2
    • Grace (prefers mornings) [Staff, max 6/period, rest ≥ 10:00:00]
        ↳ Preferences: 7 shift(s), weight 3
    • Heidi (contractor, no preferences) [Contractor, max 5/period, rest ≥ 8:00:00]

    Shifts:
    • Day 1 Shift 1: Wed 2025-01-01 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 1 Shift 2: Wed 2025-01-01 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 1 Shift 3: Wed 2025-01-01 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 2 Shift 1: Thu 2025-01-02 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 2 Shift 2: Thu 2025-01-02 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 2 Shift 3: Thu 2025-01-02 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 3 Shift 1: Fri 2025-01-03 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 3 Shift 2: Fri 2025-01-03 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 3 Shift 3: Fri 2025-01-03 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 4 Shift 1: Sat 2025-01-04 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 4 Shift 2: Sat 2025-01-04 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 4 Shift 3: Sat 2025-01-04 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 5 Shift 1: Sun 2025-01-05 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 5 Shift 2: Sun 2025-01-05 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 5 Shift 3: Sun 2025-01-05 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 6 Shift 1: Mon 2025-01-06 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 6 Shift 2: Mon 2025-01-06 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 6 Shift 3: Mon 2025-01-06 16:00 — 00:00 (Demand: 1, Period: 0)
    • Day 7 Shift 1: Tue 2025-01-07 00:00 — 08:00 (Demand: 1, Period: 0)
    • Day 7 Shift 2: Tue 2025-01-07 08:00 — 16:00 (Demand: 2, Period: 0)
    • Day 7 Shift 3: Tue 2025-01-07 16:00 — 00:00 (Demand: 1, Period: 0)

    === Solution ===
    - Objective value: -70
    - Timestamp: 2025-07-16 20:46:19

    Assignments:
    • Alice (prefers mornings, no Sundays): 6 shift(s)
        ↳ Day 1 Shift 1 (Wed 00:00)
        ↳ Day 2 Shift 1 (Thu 00:00)
        ↳ Day 3 Shift 1 (Fri 00:00)
        ↳ Day 4 Shift 1 (Sat 00:00)
        ↳ Day 6 Shift 1 (Mon 00:00)
        ↳ Day 7 Shift 1 (Tue 00:00)
    • Bob (prefers nights): 6 shift(s)
        ↳ Day 1 Shift 3 (Wed 16:00)
        ↳ Day 3 Shift 3 (Fri 16:00)
        ↳ Day 4 Shift 3 (Sat 16:00)
        ↳ Day 5 Shift 3 (Sun 16:00)
        ↳ Day 6 Shift 3 (Mon 16:00)
        ↳ Day 7 Shift 3 (Tue 16:00)
    • Clara (prefers weekends, blocks weekdays): 2 shift(s)
        ↳ Day 4 Shift 2 (Sat 08:00)
        ↳ Day 5 Shift 2 (Sun 08:00)
    • Dan (no preferences): 4 shift(s)
        ↳ Day 1 Shift 2 (Wed 08:00)
        ↳ Day 2 Shift 3 (Thu 16:00)
        ↳ Day 3 Shift 2 (Fri 08:00)
        ↳ Day 7 Shift 2 (Tue 08:00)
    • Eve (contractor, prefers day shifts): 3 shift(s)
        ↳ Day 1 Shift 2 (Wed 08:00)
        ↳ Day 2 Shift 2 (Thu 08:00)
        ↳ Day 6 Shift 2 (Mon 08:00)
    • Frank (prefers day shifts): 6 shift(s)
        ↳ Day 2 Shift 2 (Thu 08:00)
        ↳ Day 3 Shift 2 (Fri 08:00)
        ↳ Day 4 Shift 2 (Sat 08:00)
        ↳ Day 5 Shift 2 (Sun 08:00)
        ↳ Day 6 Shift 2 (Mon 08:00)
        ↳ Day 7 Shift 2 (Tue 08:00)
    • Grace (prefers mornings): 6 shift(s)
        ↳ Day 1 Shift 1 (Wed 00:00)
        ↳ Day 2 Shift 1 (Thu 00:00)
        ↳ Day 4 Shift 1 (Sat 00:00)
        ↳ Day 5 Shift 1 (Sun 00:00)
        ↳ Day 6 Shift 1 (Mon 00:00)
        ↳ Day 7 Shift 1 (Tue 00:00)
    • Heidi (contractor, no preferences): 0 shift(s)
    """

    base_date = datetime(2025, 1, 1)
    shift_length = 8
    shifts = []

    # 7 days, 3 shifts/day = 21 shifts
    for day in range(7):
        for idx, hour in enumerate([0, 8, 16]):  # Morning, Day, Night
            start = base_date + timedelta(days=day, hours=hour)
            end = start + timedelta(hours=shift_length)
            shifts.append(
                Shift(
                    name=f"Day {day + 1} Shift {idx + 1}",
                    start_time=start,
                    end_time=end,
                    demand=2 if hour == 8 else 1,  # Higher demand for day shifts
                )
            )

    # Total demand: 7*2 (day) + 7*1 (morning) + 7*1 (night) = 28 + 7 + 7 = 42 shifts needed

    # 8 nurses * 6 shifts max = 48 shifts possible ⇒ feasible

    nurses = [
        Nurse(
            name="Alice (prefers mornings, no Sundays)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 0},
            blocked_shifts={s.uid for s in shifts if s.start_time.weekday() == 6},
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Bob (prefers nights)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 16},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Clara (prefers weekends, blocks weekdays)",
            preferred_shifts={
                s.uid for s in shifts if s.start_time.weekday() in {5, 6}
            },
            blocked_shifts={s.uid for s in shifts if s.start_time.weekday() < 5},
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Dan (no preferences)",
            preferred_shifts=set(),
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=8),
            preferred_shift_weight=1,
        ),
        Nurse(
            name="Eve (contractor, prefers day shifts)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 8},
            blocked_shifts=set(),
            staff=False,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Frank (prefers day shifts)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 8},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=2,
        ),
        Nurse(
            name="Grace (prefers mornings)",
            preferred_shifts={s.uid for s in shifts if s.start_time.hour == 0},
            blocked_shifts=set(),
            staff=True,
            min_time_between_shifts=timedelta(hours=10),
            preferred_shift_weight=3,
        ),
        Nurse(
            name="Heidi (contractor, no preferences)",
            preferred_shifts=set(),
            blocked_shifts=set(),
            staff=False,
            min_time_between_shifts=timedelta(hours=8),
            preferred_shift_weight=1,
        ),
    ]

    instance = NurseRosteringInstance(
        nurses=nurses, shifts=sorted(shifts, key=lambda s: s.start_time), staff_weight=2
    )

    model = NurseRosteringModel(instance)
    solution = model.solve()
    assert solution is not None, "The solution should not be None"
    assert_solution_is_feasible(instance, solution)
    print_instance_and_solution(instance, solution)
