from ortools.sat.python import cp_model


def test_lin_constraints():
    model = cp_model.CpModel()
    x = model.new_int_var(-100, 100, "x")
    y = model.new_int_var(-100, 100, "y")
    z = model.new_int_var(-100, 100, "z")

    # abs_xz == |x+z|
    abs_xz = model.new_int_var(0, 200, "|x+z|")  # ub = ub(x)+ub(z)
    model.add_abs_equality(target=abs_xz, expr=x + z)
    # max_xyz = max(x,y,z-1)
    max_xyz = model.new_int_var(0, 100, "max(x,y, z-1)")
    model.add_max_equality(target=max_xyz, exprs=[x, y, z - 1])
    # min_xyz = min(x,y,z)
    min_xyz = model.new_int_var(-100, 100, " min(x,y, z)")
    model.add_min_equality(target=min_xyz, exprs=[x, y, z])
