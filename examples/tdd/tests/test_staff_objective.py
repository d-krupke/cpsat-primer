from nurserostering.modules import PreferStaffModule, DemandSatisfactionModule
from cpsat_utils.testing import assert_objective
from _generate import create_shifts, create_nurse

from nurserostering.nurse_vars import NurseDecisionVars
from nurserostering.data_schema import NurseRosteringInstance

from ortools.sat.python import cp_model


def test_prefer_staff_module():
    """
    Prefer assigning the staff nurse (objective = 0).
    """
    shifts = create_shifts(1)
    shifts[0].demand = 1

    staff = create_nurse("Staff", staff=True)
    contractor = create_nurse("Contractor", staff=False)
    instance = NurseRosteringInstance(nurses=[staff, contractor], shifts=shifts)

    model = cp_model.CpModel()
    vars_staff = NurseDecisionVars(staff, shifts, model)
    vars_contractor = NurseDecisionVars(contractor, shifts, model)
    solver = cp_model.CpSolver()
    staff_mod = PreferStaffModule()
    DemandSatisfactionModule().build(instance, model, [vars_staff, vars_contractor])

    model.minimize(staff_mod.build(instance, model, [vars_staff, vars_contractor]))
    assert_objective(
        model=model, solver=solver, expected=0.0
    )  # will run solve automatically
    assert solver.value(vars_staff.is_assigned_to(shifts[0].uid)) == 1, (
        "Staff nurse should be assigned to the shift"
    )
    assert solver.value(vars_contractor.is_assigned_to(shifts[0].uid)) == 0, (
        "Contractor nurse should not be assigned to the shift"
    )
