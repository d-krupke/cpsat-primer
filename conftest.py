"""
Root conftest.

Exists only so `tests/test_seo_postprocess.py` can import the auxiliary scripts
in `.scripts/`, which is not a package and not on `sys.path` by default.
Added 2026-08-06 together with `.scripts/seo_postprocess.py`. No fixtures belong here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".scripts"))
