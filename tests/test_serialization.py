import tempfile


def test_serialization():
    from ortools.sat.python import cp_model
    from ortools.sat import cp_model_pb2
    from google.protobuf import text_format
    from pathlib import Path

    def _detect_binary_mode(filename: str) -> bool:
        if filename.endswith((".txt", ".pbtxt", ".pb.txt")):
            return False
        if filename.endswith((".pb", ".bin", ".proto.bin", ".dat")):
            return True
        raise ValueError(f"Unknown extension for file: {filename}")

    # Changed in ortools 9.15: was model.Proto().SerializeToString() / text_format.MessageToString()
    # Now use model.export_to_file() which auto-detects format by extension
    def export_model(
        model: cp_model.CpModel, filename: str, binary: bool | None = None
    ):
        binary = _detect_binary_mode(filename) if binary is None else binary
        # export_to_file uses .txt extension for text format, otherwise binary
        # So we need to handle the mismatch for some extensions
        if binary and filename.endswith(".txt"):
            # Force binary even with .txt extension - use temp file
            temp_file = filename + ".pb"
            model.export_to_file(temp_file)
            Path(filename).write_bytes(Path(temp_file).read_bytes())
            Path(temp_file).unlink()
        elif not binary and not filename.endswith(".txt"):
            # Force text even without .txt extension
            temp_file = filename + ".txt"
            model.export_to_file(temp_file)
            Path(filename).write_text(Path(temp_file).read_text())
            Path(temp_file).unlink()
        else:
            model.export_to_file(filename)

    # Changed in ortools 9.15: was model.Proto().ParseFromString() / text_format.Parse()
    # Now use model.Proto().parse_text_format() for text, or cp_model_pb2 for binary
    def import_model(filename: str, binary: bool | None = None) -> cp_model.CpModel:
        binary = _detect_binary_mode(filename) if binary is None else binary
        model = cp_model.CpModel()
        if binary:
            # Parse binary via standard protobuf, then convert to text for import
            proto = cp_model_pb2.CpModelProto()
            proto.ParseFromString(Path(filename).read_bytes())
            model.Proto().parse_text_format(text_format.MessageToString(proto))
        else:
            model.Proto().parse_text_format(Path(filename).read_text())
        return model

    def rename_variable_names(model: cp_model.CpModel):
        for i, var in enumerate(model.Proto().variables):
            var.name = f"x{i}"

    # Changed in ortools 9.15: Proto() objects no longer support == comparison
    # Compare via text representation instead
    def protos_equal(m1: cp_model.CpModel, m2: cp_model.CpModel) -> bool:
        with tempfile.TemporaryDirectory() as td:
            m1.export_to_file(f"{td}/m1.txt")
            m2.export_to_file(f"{td}/m2.txt")
            return Path(f"{td}/m1.txt").read_text() == Path(f"{td}/m2.txt").read_text()

    model = cp_model.CpModel()
    x = [model.NewIntVar(0, 10, f"x{i}") for i in range(10)]
    model.add(sum(x) <= 20)
    model.maximize(sum(i * x[i] for i in range(10)))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(model, f"{tmpdir}/model.pb")
        model2 = import_model(f"{tmpdir}/model.pb")
        assert protos_equal(model, model2)
        export_model(model, f"{tmpdir}/model.txt", binary=False)
        model3 = import_model(f"{tmpdir}/model.txt", binary=False)
        assert protos_equal(model, model3)
        export_model(model, f"{tmpdir}/model.pbtxt", binary=False)
        model4 = import_model(f"{tmpdir}/model.pbtxt", binary=False)
        assert protos_equal(model, model4)
        export_model(model, f"{tmpdir}/model.pb.txt", binary=True)
        model5 = import_model(f"{tmpdir}/model.pb.txt", binary=True)
        assert protos_equal(model, model5)
        rename_variable_names(model)
        export_model(model, f"{tmpdir}/model_renamed.pb.txt")
        model6 = import_model(f"{tmpdir}/model_renamed.pb.txt")
        solver = cp_model.CpSolver()
        status = solver.Solve(model6)
        assert status == cp_model.OPTIMAL
