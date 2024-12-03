import tempfile


def test_serialization():
    from ortools.sat.python import cp_model
    from google.protobuf import text_format
    from pathlib import Path

    def _detect_binary_mode(filename: str) -> bool:
        if filename.endswith((".txt", ".pbtxt", ".pb.txt")):
            return False
        if filename.endswith((".pb", ".bin", ".proto.bin", ".dat")):
            return True
        raise ValueError(f"Unknown extension for file: {filename}")

    def export_model(
        model: cp_model.CpModel, filename: str, binary: bool | None = None
    ):
        binary = _detect_binary_mode(filename) if binary is None else binary
        if binary:
            Path(filename).write_bytes(model.Proto().SerializeToString())
        else:
            Path(filename).write_text(text_format.MessageToString(model.Proto()))

    def import_model(filename: str, binary: bool | None = None) -> cp_model.CpModel:
        binary = _detect_binary_mode(filename) if binary is None else binary
        model = cp_model.CpModel()
        if binary:
            model.Proto().ParseFromString(Path(filename).read_bytes())
        else:
            text_format.Parse(Path(filename).read_text(), model.Proto())
        return model

    def rename_variable_names(model: cp_model.CpModel):
        for i, var in enumerate(model.proto.variables):
            var.name = f"x{i}"

    model = cp_model.CpModel()
    x = [model.NewIntVar(0, 10, f"x{i}") for i in range(10)]
    model.add(sum(x) <= 20)
    model.maximize(sum(i * x[i] for i in range(10)))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(model, f"{tmpdir}/model.pb")
        model2 = import_model(f"{tmpdir}/model.pb")
        assert model.Proto() == model2.Proto()
        export_model(model, f"{tmpdir}/model.txt", binary=False)
        model3 = import_model(f"{tmpdir}/model.txt", binary=False)
        assert model.Proto() == model3.Proto()
        export_model(model, f"{tmpdir}/model.pbtxt", binary=False)
        model4 = import_model(f"{tmpdir}/model.pbtxt", binary=False)
        assert model.Proto() == model4.Proto()
        export_model(model, f"{tmpdir}/model.pb.txt", binary=True)
        model5 = import_model(f"{tmpdir}/model.pb.txt", binary=True)
        assert model.Proto() == model5.Proto()
        rename_variable_names(model)
        export_model(model, f"{tmpdir}/model_renamed.pb.txt")
        model6 = import_model(f"{tmpdir}/model_renamed.pb.txt")
        solver = cp_model.CpSolver()
        status = solver.Solve(model6)
        assert status == cp_model.OPTIMAL
