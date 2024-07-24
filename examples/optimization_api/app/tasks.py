from config import get_db_connection
from solver import TspSolver
from datetime import datetime
from uuid import UUID


def run_optimization_job(job_id: UUID) -> None:
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
