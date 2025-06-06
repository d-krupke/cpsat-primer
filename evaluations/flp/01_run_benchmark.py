"""
This file runs the actual benchmark on the instances.

Slurminade: This script uses the slurminade package to distribute the benchmark on a cluster. If you do not have a slurm-cluster, it will run the benchmark locally.
AlgBench: This script uses the AlgBench package to capture and manage the results
"""

from pathlib import Path

# for saving the results easily
from algbench import Benchmark

from instance_schema import iter_all_instances  # pip install algbench


def solve(instance_name, model, time_limit, instance_):
    from solve import solve1, solve2, solve3

    solver = {
        "solve1": solve1,
        "solve2": solve2,
        "solve3": solve3,
    }
    result = solver[model](instance_, time_limit)
    return {
        "objective": result[1],  # Total cost
        "bound": result[2],  # Best objective bound
    }


# --------------------------
# Run the benchmark on all instances.
if __name__ == "__main__":
    benchmark = Benchmark("./.data", hide_output=False)
    for instance in iter_all_instances(
        Path(
            "/home/krupke/Repositories/instance_repository_backend/REPOSITORY/flp/instances/KoerkelGhosh-asym"
        )
    ):
        for model in ["solve1", "solve2", "solve3"]:
            benchmark.add(
                solve,
                instance_name=instance.instance_uid,
                instance_=instance,
                time_limit=30,
                model=model,
            )

    benchmark.compress()
