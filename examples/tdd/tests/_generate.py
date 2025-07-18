"""
A simple utility to create shifts and nurses for testing purposes.
"""

from datetime import datetime, timedelta
from nurserostering.data_schema import Shift, Nurse


def create_shifts(k: int, week: int = 0, shift_length: int = 8) -> list[Shift]:
    """
    Create a list of shifts for testing.
    Each shift is named "Shift {i}" and has a start time and end time.
    """
    shifts = []
    for i in range(k):
        start_time = (
            datetime(2025, 1, 1, 0, 0)
            + timedelta(hours=i * shift_length)
            + timedelta(days=week * 7)
        )
        end_time = (
            datetime(2025, 1, 1, 0, 0)
            + timedelta(hours=(i + 1) * shift_length)
            + timedelta(days=week * 7)
        )
        shifts.append(
            Shift(
                name=f"Shift {i + 1}",
                start_time=start_time,
                end_time=end_time,
                demand=2,
            )
        )
    return shifts


def create_nurse(
    nurse_name: str = "Test Nurse",
    preferred_shifts: set[int] | None = None,
    blocked_shifts: set[int] | None = None,
    staff: bool = True,
    max_shifts_per_period: int = 5,
    min_time_between_shifts: timedelta = timedelta(hours=8),
) -> Nurse:
    """
    Create a nurse with customizable attributes.
    """
    if preferred_shifts is None:
        preferred_shifts = set()
    if blocked_shifts is None:
        blocked_shifts = set()
    return Nurse(
        name=nurse_name,
        preferred_shifts=preferred_shifts,
        blocked_shifts=blocked_shifts,
        staff=staff,
        min_time_between_shifts=min_time_between_shifts,
    )
