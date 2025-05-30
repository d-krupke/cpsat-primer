from ortools.sat.python import cp_model
from .piecewise_linear_function import (
    PiecewiseLinearFunction,
    PiecewiseLinearConstraint,
    are_colinear,
    generate_integer_linear_expression_from_two_points,
    get_convex_envelope,
    split_into_convex_segments,
)


def test_piecewise_linear_function():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    assert round(f(0)) == 0
    assert round(f(5)) == 5
    assert round(f(10)) == 10
    assert round(f(16)) == 7
    assert round(f(20)) == 5
    assert f.is_convex()


def test_are_colinear():
    assert are_colinear((0, 0), (10, 10), (20, 20))
    assert are_colinear((0, 1), (10, 11), (20, 21))
    assert not are_colinear((0, 0), (10, 10), (20, 21))
    assert not are_colinear((0, 0), (10, 10), (20, 19))


def test_split_into_convex_upper_bound_segments():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    parts = split_into_convex_segments(f, upper_bound=True)
    assert len(parts) == 2
    assert all(p.is_convex(True) for p in parts)
    assert all(parts[0](x) == f(x) for x in range(10))
    assert all(parts[1](x) == f(x) for x in range(10, 21))
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    parts = split_into_convex_segments(f, upper_bound=True)
    assert len(parts) == 1
    assert all(p.is_convex(True) for p in parts)
    assert all(p(0) == f(0) for p in parts)
    assert all(p(10) == f(10) for p in parts)
    assert all(p(20) == f(20) for p in parts)


def test_generate_integer_linear_expression():
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, 10) == (1, 1, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 20, 10) == (2, 1, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, 15) == (2, 3, 0)
    assert generate_integer_linear_expression_from_two_points(0, 0, 10, -10) == (
        1,
        -1,
        0,
    )
    assert generate_integer_linear_expression_from_two_points(-10, -10, 10, 10) == (
        1,
        1,
        0,
    )


def test_get_upper_bounding_convex_envelope():
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    g = get_convex_envelope(f, upper_bound=True)
    assert g.is_convex()
    assert all(g(x) >= f(x) for x in range(21))
    assert g(0) == 0
    assert g(10) == 25
    assert g(20) == 50
    g_ = get_convex_envelope(f, upper_bound=False)
    assert g_.is_convex(upper_bound=False)
    assert all(g_(x) <= f(x) for x in range(21))
    assert g_(0) == 0
    assert g_(10) == 10
    assert g_(20) == 50


def test_piecewise_linear_upper_bound_constraint():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.maximize(c.y)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c.y) == 10
    assert solver.value(x) == 10
    assert c.num_auxiliary_variables == 0
    assert c.num_constraints == 2

    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.maximize(c.y)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c.y) == 50
    assert solver.value(x) == 20
    assert c.num_auxiliary_variables == 2

    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 50])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=False)
    model.minimize(c.y)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c.y) == 0
    assert solver.value(x) == 0

    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20, 30], ys=[20, 10, 50, 40])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=False)
    model.minimize(c.y)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c.y) == 10
    assert solver.value(x) == 10

    # Test with zero-gradient segments
    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 10])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.maximize(c.y)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c.y) == 10
    assert solver.value(x) == 10
