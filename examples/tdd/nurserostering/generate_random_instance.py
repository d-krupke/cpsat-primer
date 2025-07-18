"""
This module just provides some basic code for generating random instances of the nurse rostering problem.
"""

import random
from datetime import datetime, timedelta
from .data_schema import Nurse, Shift, NurseRosteringInstance


class NursePreferenceGenerator:
    def __init__(
        self,
        shifts: list[Shift],
        preference_probability: float = 0.3,
        block_probability: float = 0.05,
    ):
        self.shifts = shifts
        self.preference_probability = preference_probability
        self.block_probability = block_probability

        self.morning_uids = {s.uid for s in shifts if s.start_time.hour == 0}
        self.afternoon_uids = {s.uid for s in shifts if s.start_time.hour == 8}
        self.night_uids = {s.uid for s in shifts if s.start_time.hour == 16}
        self.weekend_uids = {s.uid for s in shifts if s.start_time.weekday() >= 5}

    def assign(self, preference_type: str) -> tuple[set[int], set[int]]:
        preferred = set()
        blocked = set()

        if preference_type == "morning":
            preferred = self._sample(self.morning_uids)
        elif preference_type == "afternoon":
            preferred = self._sample(self.afternoon_uids)
        elif preference_type == "night":
            preferred = self._sample(self.night_uids)
        elif preference_type == "weekend":
            preferred = self._sample(self.weekend_uids)
        elif preference_type == "none":
            preferred = self._sample({s.uid for s in self.shifts})

        blocked_candidates = set(s.uid for s in self.shifts if s.uid not in preferred)
        blocked = {
            uid
            for uid in blocked_candidates
            if random.random() < self.block_probability
        }
        preferred = (
            preferred - blocked
        )  # Ensure no overlap between preferred and blocked shifts

        return preferred, blocked

    def _sample(self, uids: set[int]) -> set[int]:
        return {uid for uid in uids if random.random() < self.preference_probability}


def generate_random_instance(
    num_nurses: int = 10,
    num_days: int = 7,
    staff_ratio: float = 0.7,
    min_shift_demand: int = 1,
    max_shift_demand: int = 3,
    rest_hours: int = 11,
    base_date: datetime = datetime(2025, 1, 1),
) -> NurseRosteringInstance:
    shifts = []
    shift_length = 8
    shift_hours = [0, 8, 16]
    for day in range(num_days):
        for idx, h in enumerate(shift_hours):
            start = base_date + timedelta(days=day, hours=h)
            end = start + timedelta(hours=shift_length)
            shift = Shift(
                name=f"{start.strftime('%Y-%m-%d')} Shift {idx + 1}",
                start_time=start,
                end_time=end,
                demand=random.randint(min_shift_demand, max_shift_demand),
            )
            shifts.append(shift)

    preference_gen = NursePreferenceGenerator(shifts)

    nurses = []
    preference_types = ["morning", "afternoon", "night", "weekend", "none"]
    for i in range(num_nurses):
        staff = random.random() < staff_ratio
        rest_time = timedelta(hours=rest_hours)

        pref_type = random.choice(preference_types)
        preferred, blocked = preference_gen.assign(pref_type)

        nurse = Nurse(
            name=f"Nurse {i + 1}",
            preferred_shifts=preferred,
            blocked_shifts=blocked,
            staff=staff,
            min_time_between_shifts=rest_time,
        )
        nurses.append(nurse)

    return NurseRosteringInstance(
        nurses=nurses, shifts=sorted(shifts, key=lambda s: s.start_time)
    )
