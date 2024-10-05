"""
This file runs the actual benchmark on the instances.

Slurminade: This script uses the slurminade package to distribute the benchmark on a cluster. If you do not have a slurm-cluster, it will run the benchmark locally.
AlgBench: This script uses the AlgBench package to capture and manage the results
"""

import logging
import random

# for saving the results easily
from algbench import Benchmark  # pip install algbench

# for distributing the benchmark on a cluster
import slurminade  # pip install slurminade
from _utils import TspLibGraphInstanceDb
from _conf import (
    INSTANCE_DB,
    EXPERIMENT_DATA,
    TIME_LIMIT,
    STRATEGIES,
    OPTIMALITY_TOLERANCES,
)

instances = TspLibGraphInstanceDb(INSTANCE_DB)
benchmark = Benchmark(EXPERIMENT_DATA)

# -----------------------------------------
# Distribution configuration for Slurm
# If you don't have Slurm, this won't do anything.
# If you have slurm, you have to update the configuration to your needs.
slurminade.update_default_configuration(
    # This setup is for the TU BS Alg cluster.
    # This doubles as documentation for on which cluster the benchmark was run.
    partition="alg",
    constraint="alggen05",
    cpus_per_task=24,
    mail_user="...@...",
    mail_type="FAIL",  # Send a mail if a job fails.
)
slurminade.set_dispatch_limit(1_000)
# -----------------------------------------


@slurminade.slurmify()  # makes the function distributable on a cluster
def load_instance_and_run_solver(instance_name):
    try:
        instance = instances[instance_name]
        if instance.number_of_nodes() == 0:
            return
    except ValueError:
        return

    # The logging framework is much more powerful than print statements.
    # I recommend using it instead of print statements to report progress.
    logger = logging.getLogger("Evaluation")
    benchmark.capture_logger("Evaluation", logging.INFO)

    def run_solver(instance_name, time_limit, strategy, opt_tol, _instance):
        # Arguments starting with _ are not saved in the experiment data.
        # The instance is already in the instance database.
        # We only need the instance name to compare the results.

        solver = STRATEGIES[strategy](_instance, logger=logger)
        obj, lb = solver.solve(time_limit, opt_tol)
        return {
            "num_nodes": _instance.number_of_nodes(),
            "lower_bound": lb,
            "objective": obj,
        }

    # Will only run if the instance is not already solved.
    for strategy in STRATEGIES:
        for opt_tol in OPTIMALITY_TOLERANCES:
            benchmark.add(
                run_solver, instance_name, TIME_LIMIT, strategy, opt_tol, instance
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
    instances.download()
    instance_names = list(instance for instance in instances.selection(0, 500))
    # shuffle the instances to distribute the load more evenly.
    random.shuffle(instance_names)
    # Distribute the benchmark on a cluster.
    with slurminade.Batch(max_size=2) as batch:
        for instance_name in instance_names:
            load_instance_and_run_solver.distribute(instance_name)
        # compress the results at the end.
        job_ids = batch.flush()
        compress.wait_for(job_ids).distribute()
