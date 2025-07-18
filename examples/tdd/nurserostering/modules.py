import abc
from ortools.sat.python import cp_model
from .data_schema import NurseRosteringInstance
from .nurse_vars import NurseDecisionVars


class ShiftAssignmentModule(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        """
        Add constraints and optionally return a sub-objective expression.
        Each subclass defines one constraint or objective aspect.
        """
        return 0


class NoBlockedShiftsModule(ShiftAssignmentModule):
    """
    Prohibit assignment to blocked shifts.
    """

    def enforce_for_nurse(self, model: cp_model.CpModel, nurse_x: NurseDecisionVars):
        for shift_uid in nurse_x.nurse.blocked_shifts:
            # prohibit assignment to blocked shifts
            model.add(nurse_x.is_assigned_to(shift_uid) == 0)

    def build(
        self,
        instance: NurseRosteringInstance,
        model: cp_model.CpModel,
        nurse_shift_vars: list[NurseDecisionVars],
    ) -> cp_model.LinearExprT:
        for nurse_x in nurse_shift_vars:
            self.enforce_for_nurse(model, nurse_x)
        return 0


class DemandSatisfactionModule(ShiftAssignmentModule):
    def build(self, instance, model, nurse_shift_vars):
        """
        Ensure each shift meets its demand.
        """
        for shift in instance.shifts:
            assigned = [
                nv.is_assigned_to(shift.uid)
                for nv in nurse_shift_vars
                if shift.uid in nv._x
            ]
            model.add(sum(assigned) >= shift.demand)
        return 0


class MinTimeBetweenShifts(ShiftAssignmentModule):
    def build(self, instance, model, nurse_shift_vars):
        """
        Enforce minimum rest time between any two shifts for a nurse.
        """
        for nv in nurse_shift_vars:
            for i in range(len(nv.shifts) - 1):
                shift_i = nv.shifts[i]
                colliding = []
                for j in range(i + 1, len(nv.shifts)):
                    shift_j = nv.shifts[j]
                    if (
                        shift_i.end_time + nv.nurse.min_time_between_shifts
                        <= shift_j.start_time
                    ):
                        break
                    colliding.append(shift_j)
                if colliding:
                    model.add(
                        sum(nv.is_assigned_to(s.uid) for s in colliding) == 0
                    ).only_enforce_if(nv.is_assigned_to(shift_i.uid))
        return 0


class MaximizePreferences(ShiftAssignmentModule):
    def build(self, instance, model, nurse_shift_vars):
        """
        Encourage assigning nurses to their preferred shifts.
        Each preference counts negatively toward the minimization objective.
        """
        expr = 0
        for nv in nurse_shift_vars:
            for uid in nv.nurse.preferred_shifts:
                expr += -nv.nurse.preferred_shift_weight * nv.is_assigned_to(uid)
        return expr


class PreferStaffModule(ShiftAssignmentModule):
    def build(self, instance, model, nurse_shift_vars):
        """
        Penalize use of non-staff (contract) nurses in the objective.
        """
        expr = 0
        for nv in nurse_shift_vars:
            if not nv.nurse.staff:
                for uid in nv._x:
                    expr += instance.staff_weight * nv.is_assigned_to(uid)
        return expr
