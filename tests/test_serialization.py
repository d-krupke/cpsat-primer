"""Round-trip models through the chapter's `export_model`/`import_model`."""

import tempfile
from pathlib import Path

import pytest
from _snippets import run_snippet
from ortools.sat.python import cp_model

SERIALIZATION = run_snippet("parameters/serialization")
export_model = SERIALIZATION["export_model"]
import_model = SERIALIZATION["import_model"]


def build_model() -> cp_model.CpModel:
    model = cp_model.CpModel()
    x = [model.new_int_var(0, 10, f"x{i}") for i in range(10)]
    model.add(sum(x) <= 20)
    model.maximize(sum(i * x[i] for i in range(10)))
    return model


# Changed in ortools 9.15: Proto() objects no longer support == comparison
# Compare via text representation instead
def protos_equal(m1: cp_model.CpModel, m2: cp_model.CpModel) -> bool:
    with tempfile.TemporaryDirectory() as td:
        m1.export_to_file(f"{td}/m1.txt")
        m2.export_to_file(f"{td}/m2.txt")
        return Path(f"{td}/m1.txt").read_text() == Path(f"{td}/m2.txt").read_text()


@pytest.mark.parametrize(
    "filename, binary",
    [
        ("model.pb", None),  # binary, detected from the extension
        ("model.txt", False),
        ("model.pbtxt", False),
        ("model.pb.txt", True),  # forced binary despite the text extension
    ],
)
def test_round_trip(tmp_path, filename, binary):
    """Exporting and importing reproduces the model in every supported format."""
    model = build_model()
    export_model(model, str(tmp_path / filename), binary=binary)
    assert protos_equal(model, import_model(str(tmp_path / filename), binary=binary))


def test_renamed_model_still_solves(tmp_path):
    """Anonymizing variable names (e.g., before sharing a model) keeps it valid."""
    model = build_model()
    for i, var in enumerate(model.Proto().variables):
        var.name = f"x{i}"
    export_model(model, str(tmp_path / "model_renamed.pb.txt"))
    model = import_model(str(tmp_path / "model_renamed.pb.txt"))
    assert cp_model.CpSolver().solve(model) == cp_model.OPTIMAL


def test_unknown_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown extension"):
        export_model(build_model(), str(tmp_path / "model.json"))
