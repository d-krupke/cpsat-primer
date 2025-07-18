from nurserostering.modules import MaximizePreferences, DemandSatisfactionModule
from cpsat_utils.testing import assert_objective
from _generate import create_shifts, create_nurse

from nurserostering.nurse_vars import NurseDecisionVars
from nurserostering.data_schema import NurseRosteringInstance

from ortools.sat.python import cp_model


def test_maximize_preferences_module():
    """
    Prefer assigning the nurse to their preferred shift (objective = -1).
    """
    shifts = create_shifts(1)
    shifts[0].demand = 1

    nurse = create_nurse("Preferred Nurse", preferred_shifts={shifts[0].uid})
    instance = NurseRosteringInstance(nurses=[nurse], shifts=shifts)

    model = cp_model.CpModel()
    nurse_vars = NurseDecisionVars(nurse, shifts, model)
    solver = cp_model.CpSolver()
    pref_mod = MaximizePreferences()

    DemandSatisfactionModule().build(instance, model, [nurse_vars])

    model.minimize(pref_mod.build(instance, model, [nurse_vars]))

    assert_objective(model=model, solver=solver, expected=-1.0)

    assert solver.value(nurse_vars.is_assigned_to(shifts[0].uid)) == 1, (
        "Nurse should be assigned to their preferred shift"
    )
