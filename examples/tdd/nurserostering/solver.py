from ortools.sat.python import cp_model
from .nurse_vars import NurseDecisionVars
from .data_schema import NurseRosteringInstance, NurseRosteringSolution
from .modules import (
    ShiftAssignmentModule,
    NoBlockedShiftsModule,
    DemandSatisfactionModule,
    MinTimeBetweenShifts,
    MaximizePreferences,
    PreferStaffModule,
)


class NurseRosteringModel:
    """
    A compact and extensible solver for the nurse rostering problem using CP-SAT.
    """

    def __init__(
        self, instance: NurseRosteringInstance, model: cp_model.CpModel | None = None
    ):
        self.instance = instance
        self.model = model or cp_model.CpModel()
        self.nurse_vars = [
            NurseDecisionVars(nurse, instance.shifts, self.model)
            for nurse in instance.nurses
        ]

        self.modules: list[ShiftAssignmentModule] = [
            NoBlockedShiftsModule(),
            DemandSatisfactionModule(),
            MinTimeBetweenShifts(),
            MaximizePreferences(),
            PreferStaffModule(),
        ]

        objective = sum(
            module.build(instance, self.model, self.nurse_vars)  # type: ignore
            for module in self.modules
        )
        self.model.minimize(objective)

    def solve(
        self,
        log_search_progress: bool = True,
        max_time_in_seconds: float = 60.0,
        **solver_params,
    ) -> NurseRosteringSolution:
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = log_search_progress
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        for key, value in solver_params.items():
            setattr(solver.parameters, key, value)

        status = solver.solve(self.model)
        if status == cp_model.INFEASIBLE:
            raise ValueError("The model is infeasible.")
        elif status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("Solver failed to find a feasible solution.")

        nurses_at_shifts = {}
        for nurse_model in self.nurse_vars:
            for shift_uid in nurse_model.extract(solver):
                nurses_at_shifts.setdefault(shift_uid, []).append(nurse_model.nurse.uid)

        return NurseRosteringSolution(
            nurses_at_shifts=nurses_at_shifts,
            objective_value=round(solver.objective_value),
        )
