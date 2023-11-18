"""
! YOU SHOULD ACTUALLY NOT HAVE YOUR IMPLEMENTATIONS IN YOUR EVALUATION !
Implement your solvers in proper packages that you can import into your evaluation.
Keeping track of the git revision of your solver is sufficient.
If you have multiple benchmarks, the risk of accidentally using outdated versions of your solver is high.
If you just want to do some quick exploration and don't expect the solver to be used anywhere else, you can put it in your evaluation.
"""

from .mip import GurobiTspSolver
from .cpsat_v1 import CpSatTspSolverV1
from .cpsat_v2 import CpSatTspSolverDantzig
from .cpsat_v3 import CpSatTspSolverMtz

__all__ = ['GurobiTspSolver', 'CpSatTspSolverV1', 'CpSatTspSolverDantzig', 'CpSatTspSolverMtz']