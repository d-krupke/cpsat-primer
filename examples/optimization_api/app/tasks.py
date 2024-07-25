"""
This file is responsible for running the optimization job in a separate worker.
"""

from config import get_db_connection
from models import TspJobRequest, TspJobStatus
from solver import TspSolver
from datetime import datetime
from uuid import UUID
from db import TspJobDbConnection
import httpx
import logging


def send_webhook(job_request: TspJobRequest, job_status: TspJobStatus) -> None:
    if job_request.webhook_url:
        try:
            # Send a POST request to the webhook URL
            response = httpx.post(
                url=f"{job_request.webhook_url}", json=job_status.model_dump_json()
            )
            response.raise_for_status()  # Raise an error for bad responses
        except httpx.HTTPStatusError as e:
            logging.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}")


def run_optimization_job(
    job_id: UUID, db_connection: TspJobDbConnection | None = None
) -> None:
    """
    Will fetch the job request from the database, run the optimization algorithm,
    and store the solution back in the database. Finally, it will send a webhook
    to the URL specified in the job. This function may be run on a separate worker,
    which is why we do not pass or return data directly, but rather use the database.
    """
    if db_connection is None:
        db_connection = get_db_connection()
    job_status = db_connection.get_status(job_id)
    job_request = db_connection.get_request(job_id)
    if job_status is None or job_request is None:
        return  # job got deleted
    job_status.status = "Running"
    job_status.started_at = datetime.now()
    db_connection.update_job_status(job_status)
    solver = TspSolver(job_request.tsp_instance, job_request.optimization_parameters)
    solution = solver.solve(log_callback=print)
    db_connection.set_solution(job_id, solution)
    job_status.status = "Completed"
    job_status.completed_at = datetime.now()
    db_connection.update_job_status(job_status)
    send_webhook(job_request, job_status)


if __name__ == "__main__":
    # allow calling the algorithm via command line so it can be run by an
    # external task queue, based on commands, e.g., `python solver.py <task_id>`

    import sys

    # the command could look like this `some_task_queue run python solver.py <task_id>`
    # so, we simply try to parse the last argument as a UUID.
    task_id = UUID(sys.argv[-1])
    run_optimization_job(task_id)
