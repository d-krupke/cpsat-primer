from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "metrics", [[("runtime", "min")], [("runtime", "min"), ("runtime", "min")]]
)
def test_comparison_grid_pairs_same_instances(metrics):
    chapter = (
        Path(__file__).resolve().parents[1] / "chapters/benchmarking.md"
    ).read_text(encoding="utf-8")
    code = re.findall(r"```python\n(.*?)```", chapter, re.S)[0]
    namespace = {}
    exec(code, namespace)
    baseline = pd.DataFrame({"runtime": [1.0, 2.0]}, index=["a", "b"])
    reordered = baseline.iloc[::-1]
    fig, axes = namespace["plot_comparison_grid"](baseline, reordered, metrics)
    try:
        assert axes[0].collections[0].get_offsets().tolist() == [[1.0, 1.0], [2.0, 2.0]]
    finally:
        plt.close(fig)
