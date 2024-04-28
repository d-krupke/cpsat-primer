from ortools.sat.python import cp_model

def test_simple_intvar_example():
   model = cp_model.CpModel()

   x = model.NewIntVar(0, 100, "x")
   y = model.NewIntVar(0, 100, "y")

   model.Add(x + y <= 30)
   model.Maximize(30 * x + 50 * y)

   solver = cp_model.CpSolver()
   status = solver.Solve(model)
   assert status == cp_model.OPTIMAL, "Should be optimal"

   assert solver.Value(x) == 0, "x is expected to be 0"
   assert solver.Value(y) == 30, "y is expected to be 30"