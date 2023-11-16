"""
Having a separate file for the configuration of paths and constants allows
you to, e.g., quickly change the database without having to overwrite the
old data.
"""
from pathlib import Path
from solver import GurobiTspSolver, CpSatTspSolverV1

# Data that is meant to be shared to verify the results.
PUBLIC_DATA = Path(__file__).parent / "PUBLIC_DATA"
# Data meant for debugging and investigation, not to be shared because of its size.
PRIVATE_DATA = Path(__file__).parent / "PRIVATE_DATA"

# Saving the instances to repeat the experiment on exactly the same data.
INSTANCE_DB = PUBLIC_DATA / "instance_db.json"
# Saving the full experiment data for potential debugging.
EXPERIMENT_DATA = PRIVATE_DATA / "full_experiment_data"
# Saving the simplified experiment data for analysis.
SIMPLIFIED_RESULTS = PUBLIC_DATA / "simplified_results.json.zip"

# Benchmark Setup
TIME_LIMIT = 90
STRATEGIES = solvers = {
    "GurobiTspSolver": GurobiTspSolver,
    "CpSatTspSolverV1": CpSatTspSolverV1,
}
