import inspect
from ortools.sat.python import cp_model

def test_FixedSizedIntervalVar_argument_names():
    model = cp_model.CpModel()
    FixedSizedIntervalVar_signature = inspect.signature(model.NewFixedSizedIntervalVar)

    # Define the expected argument names
    expected_argument_names = ['start', 'size', 'name']

    # Get the actual argument names
    actual_argument_names = list(FixedSizedIntervalVar_signature.parameters.keys())

    # Check if the actual argument names match the expected argument names
    assert actual_argument_names == expected_argument_names, \
    f"Expected argument names {expected_argument_names}, but got {actual_argument_names}"