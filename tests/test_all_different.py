from ortools.sat.python import cp_model


def test_all_different_usage():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")

    model.add_all_different([x, y, z])
    model.add_all_different(x, y, z)

    # fancier usage including transformations
    vars = [model.new_int_var(0, 10, f"v_{i}") for i in range(10)]
    model.add_all_different(x + i for i, x in enumerate(vars))
