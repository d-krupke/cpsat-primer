"""
This file runs the actual benchmark on the instances.

Slurminade: This script uses the slurminade package to distribute the benchmark on a cluster. If you do not have a slurm-cluster, it will run the benchmark locally.
AlgBench: This script uses the AlgBench package to capture and manage the results
"""

import random
from pathlib import Path

# for saving the results easily
from algbench import Benchmark  # pip install algbench

# for distributing the benchmark on a cluster
import slurminade  # pip install slurminade

from solver import (
    RectanglePackingWithoutRotationsModel,
    RectanglePackingWithRotationsModel,
    RectangleKnapsackWithRotationsModel,
    RectangleKnapsackWithoutRotationsModel,
    Instance,
)


benchmark = Benchmark("./PRIVATE_DATA/results")

# -----------------------------------------
# Distribution configuration for Slurm
# If you don't have Slurm, this won't do anything.
# If you have slurm, you have to update the configuration to your needs.
slurminade.update_default_configuration(
    # This setup is for the TU BS Alg cluster.
    # This doubles as documentation for on which cluster the benchmark was run.
    partition="alg",
    constraint="alggen05",
    exclusive=True,
    mail_user="krupke@ibr.cs.tu-bs.de",
    mail_type="FAIL",  # Send a mail if a job fails.
)
slurminade.set_dispatch_limit(1_000)
# -----------------------------------------


def run_RectanglePackingWithoutRotationsModel(instance_name, time_limit):
    with open(Path("./instances") / instance_name, "r") as file:
        instance = Instance.model_validate_json(file.read())
    model = RectanglePackingWithoutRotationsModel(instance)
    model.solve(time_limit)
    return {
        "solution": model.solution,
        "status": model.status,
        "feasible": model.is_feasible(),
        "infeasible": model.is_infeasible(),
    }


@slurminade.slurmify()  # makes the function distributable on a cluster
def run_RectanglePackingWithoutRotationsModel_distributed(instance_name, time_limit):
    benchmark.add(run_RectanglePackingWithoutRotationsModel, instance_name, time_limit)


def run_RectanglePackingWithRotationsModel(instance_name, time_limit):
    with open(Path("./instances") / instance_name, "r") as file:
        instance = Instance.model_validate_json(file.read())
    model = RectanglePackingWithRotationsModel(instance)
    model.solve(time_limit)
    return {
        "solution": model.solution,
        "status": model.status,
        "feasible": model.is_feasible(),
        "infeasible": model.is_infeasible(),
    }


@slurminade.slurmify()  # makes the function distributable on a cluster
def run_RectanglePackingWithRotationsModel_distributed(instance_name, time_limit):
    benchmark.add(run_RectanglePackingWithRotationsModel, instance_name, time_limit)


def run_RectangleKnapsackWithRotationsModel(instance_name, time_limit, opt_tol):
    with open(Path("./instances") / instance_name, "r") as file:
        instance = Instance.model_validate_json(file.read())
    model = RectangleKnapsackWithRotationsModel(instance)
    model.solve(time_limit, opt_tol=opt_tol)
    return {
        "solution": model.solution,
        "status": model.status,
        "upper_bound": model.upper_bound,
        "objective_value": model.objective_value,
    }


@slurminade.slurmify()  # makes the function distributable on a cluster
def run_RectangleKnapsackWithRotationsModel_distributed(
    instance_name, time_limit, opt_tol
):
    benchmark.add(
        run_RectangleKnapsackWithRotationsModel, instance_name, time_limit, opt_tol
    )


def run_RectangleKnapsackWithoutRotationsModel(instance_name, time_limit, opt_tol):
    with open(Path("./instances") / instance_name, "r") as file:
        instance = Instance.model_validate_json(file.read())
    model = RectangleKnapsackWithoutRotationsModel(instance)
    model.solve(time_limit, opt_tol=opt_tol)
    return {
        "solution": model.solution,
        "status": model.status,
        "upper_bound": model.upper_bound,
        "objective_value": model.objective_value,
    }


@slurminade.slurmify()  # makes the function distributable on a cluster
def run_RectangleKnapsackWithoutRotationsModel_distributed(
    instance_name, time_limit, opt_tol
):
    benchmark.add(
        run_RectangleKnapsackWithoutRotationsModel, instance_name, time_limit, opt_tol
    )


# --------------------------
# Compression is not thread-safe so we make it a separate function
# if you only notify about failures, you may want to do
# ``@slurminade.slurmify(mail_type="ALL)`` to be notified after completion.
@slurminade.slurmify()
def compress():
    benchmark.compress()


# --------------------------
# Run the benchmark on all instances.
if __name__ == "__main__":
    instance_names = [
        str(instance_name.relative_to("./instances"))
        for instance_name in Path("./instances/hopper").iterdir()
        if instance_name.is_file() and str(instance_name)[-3:] != ".md"
    ]
    # shuffle the instances to distribute the load more evenly.
    random.shuffle(instance_names)
    # Distribute the benchmark on a cluster.
    with slurminade.Batch(max_size=5) as batch:
        for instance_name in instance_names:
            run_RectangleKnapsackWithoutRotationsModel_distributed.distribute(
                instance_name, 90.0, 0.01
            )
            run_RectangleKnapsackWithRotationsModel_distributed.distribute(
                instance_name, 90.0, 0.01
            )
            run_RectanglePackingWithoutRotationsModel_distributed.distribute(
                instance_name, 90.0
            )
            run_RectanglePackingWithRotationsModel_distributed.distribute(
                instance_name, 90.0
            )
        # compress the results at the end.
        job_ids = batch.flush()
        compress.wait_for(job_ids).distribute()
