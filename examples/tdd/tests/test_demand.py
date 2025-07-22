from nurserostering.modules import DemandSatisfactionModule
from nurserostering.nurse_vars import NurseDecisionVars
from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible
from _generate import create_shifts, create_nurse
from nurserostering.data_schema import NurseRosteringInstance


def test_demand_satisfaction_met():
    shifts = create_shifts(1)
    shifts[0].demand = 2
    nurse1 = create_nurse("N1")
    nurse2 = create_nurse("N2")
    instance = NurseRosteringInstance(nurses=[nurse1, nurse2], shifts=shifts)

    with AssertModelFeasible() as model:
        nurse_vars1 = NurseDecisionVars(nurse1, shifts, model)
        nurse_vars2 = NurseDecisionVars(nurse2, shifts, model)
        DemandSatisfactionModule().build(instance, model, [nurse_vars1, nurse_vars2])


def test_demand_satisfaction_understaffed():
    shifts = create_shifts(1)
    shifts[0].demand = 2
    nurse1 = create_nurse("N1")
    nurse2 = create_nurse("N2")
    instance = NurseRosteringInstance(nurses=[nurse1, nurse2], shifts=shifts)

    with AssertModelInfeasible() as model:
        nurse_vars1 = NurseDecisionVars(nurse1, shifts, model)
        nurse_vars2 = NurseDecisionVars(nurse2, shifts, model)
        DemandSatisfactionModule().build(instance, model, [nurse_vars1, nurse_vars2])
        nurse_vars2.fix(shifts[0].uid, False)
