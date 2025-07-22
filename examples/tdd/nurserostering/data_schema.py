"""
This module defines the data schema for the nurse rostering problem.
Note that this is just a random variant of the nurse rostering problem.

We define the instance and solution data structures using Pydantic.
"""

from datetime import datetime, timedelta
from pydantic import BaseModel, Field, NonNegativeInt, model_validator
import uuid

# Semantic type aliases for clarity
NurseUid = int
ShiftUid = int


def generate_random_uid() -> int:
    # Use uuid4 and convert to an integer (truncated to 64 bits for practical use)
    return uuid.uuid4().int >> 64


class Nurse(BaseModel):
    uid: NurseUid = Field(
        default_factory=generate_random_uid,
        description="Unique identifier for the nurse",
    )
    name: str = Field(..., description="Name of the nurse")
    preferred_shifts: set[ShiftUid] = Field(
        ..., description="List of preferred shift UIDs for the nurse"
    )
    blocked_shifts: set[ShiftUid] = Field(
        ..., description="List of blocked shift UIDs for the nurse"
    )
    staff: bool = Field(
        ...,
        description="Indicates if the nurse is a staff member (True) or a contractor (False)",
    )
    min_time_between_shifts: timedelta = Field(
        ..., description="Minimum off duty time between two shifts for the same nurse"
    )
    preferred_shift_weight: NonNegativeInt = Field(
        default=1,
        description="The weight in the objective function for every assigned preference.",
    )


class Shift(BaseModel):
    uid: ShiftUid = Field(
        default_factory=generate_random_uid,
        description="Unique identifier for the shift",
    )
    name: str = Field(..., description="Name of the shift (e.g., '2025-01-01 Morning')")
    start_time: datetime = Field(
        ..., description="Start time of the shift as a full datetime (YYYY-MM-DD HH:MM)"
    )
    end_time: datetime = Field(
        ..., description="End time of the shift as a full datetime (YYYY-MM-DD HH:MM)"
    )
    demand: NonNegativeInt = Field(
        ..., description="Number of nurses required for this shift"
    )


class NurseRosteringInstance(BaseModel):
    """
    This schema defines the INPUT for the nurse rostering problem.
    """

    nurses: list[Nurse] = Field(
        ..., description="List of nurses in the rostering instance"
    )
    shifts: list[Shift] = Field(
        ...,
        description="List of shifts that need to be covered. Shifts must be sorted in time.",
    )
    staff_weight: int = Field(
        default=1,
        description="The weight in the objective function for each assigned staff nurse.",
    )

    @model_validator(mode="after")
    def validate_shifts_unique_uids(self):
        """
        Ensure that all shifts have unique UIDs to avoid conflicts.
        """
        shift_uids = {shift.uid for shift in self.shifts}
        if len(shift_uids) != len(self.shifts):
            raise ValueError("Shift UIDs must be unique.")
        return self

    @model_validator(mode="after")
    def validate_nurses_unique_uids(self):
        """
        Ensure that all nurses have unique UIDs to avoid conflicts.
        """
        nurse_uids = {nurse.uid for nurse in self.nurses}
        if len(nurse_uids) != len(self.nurses):
            raise ValueError("Nurse UIDs must be unique.")
        return self

    @model_validator(mode="after")
    def validate_shifts_sorted_by_time(self):
        """
        Ensure that shifts are sorted by start time.
        """
        for shift_a, shift_b in zip(self.shifts, self.shifts[1:]):
            if shift_a.start_time > shift_b.start_time:
                raise ValueError("Shifts must be sorted by start time.")
        return self


class NurseRosteringSolution(BaseModel):
    """
    This schema defines the OUTPUT for the nurse rostering problem.
    """

    nurses_at_shifts: dict[ShiftUid, list[NurseUid]] = Field(
        ..., description="Maps shift UIDs to lists of assigned nurse UIDs."
    )
    objective_value: int = Field(
        description="Objective value of the computed solution."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Time when the solution was generated. Takes little space and can be extremely useful when investigating issues with the solution. Optimally, also add the revision of the algorithm that generated the solution, e.g., by using a git commit hash.",
    )
    # Validation of the solution will be handled in a separate module.
