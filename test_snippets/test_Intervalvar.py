import inspect
from ortools.sat.python import cp_model

def test_IntervalVar_argument_names():
    model = cp_model.CpModel()
    IntervalVar_signature = inspect.signature(model.NewIntervalVar)

    # Define expected argument names
    expected_argument_names = ['start', 'size', 'end', 'name']

    # Get actual argument names
    actual_argument_names = list(IntervalVar_signature.parameters.keys())

    # Check if actual argument names match expected argument names
    assert actual_argument_names == expected_argument_names, \
    f"Expected argument names {expected_argument_names}, but got {actual_argument_names}"