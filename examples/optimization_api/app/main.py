"""
This file contains the main FastAPI application.
For a larger project, we would move the routes to separate files, but for this example, we keep everything in one file.
"""

from uuid import UUID
from fastapi import FastAPI, APIRouter, HTTPException, Depends

from models import TspJobRequest, TspJobStatus
from solver import TspSolution
from config import get_db_connection, get_task_queue
from tasks import run_optimization_job

app = FastAPI(
    title="My Optimization API",
    description="This is an example on how to deploy an optimization algorithm based on CP-SAT as an API.",
)

tsp_solver_v1_router = APIRouter(tags=["TSP_solver_v1"])


@tsp_solver_v1_router.post("/jobs", response_model=TspJobStatus)
def post_job(
    job_request: TspJobRequest,
    db_connection=Depends(get_db_connection),
    task_queue=Depends(get_task_queue),
):
    """
    Submit a new job to solve a TSP instance.
    """
    job_status = db_connection.register_job(job_request)
    task_queue.enqueue(
        run_optimization_job,
        job_status.task_id,
        # adding a 60 second buffer to the job timeout
        job_timeout=job_request.optimization_parameters.timeout + 60,
    )
    return job_status


@tsp_solver_v1_router.get("/jobs/{task_id}", response_model=TspJobStatus)
def get_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the status of a job.
    """
    status = db_connection.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@tsp_solver_v1_router.get("/jobs/{task_id}/solution", response_model=TspSolution)
def get_solution(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the solution of a job, if available.
    """
    solution = db_connection.get_solution(task_id)
    if solution is None:
        raise HTTPException(status_code=404, detail="Solution not found")
    return solution


@tsp_solver_v1_router.delete("/jobs/{task_id}")
def cancel_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Deletes/cancels a job. This will *not* immediately stop the job if it is running.
    """
    db_connection.delete_job(task_id)


@tsp_solver_v1_router.get("/jobs", response_model=list[TspJobStatus])
def list_jobs(db_connection=Depends(get_db_connection)):
    """
    List all jobs.
    """
    return db_connection.list_jobs()


app.include_router(tsp_solver_v1_router, prefix="/tsp_solver/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, workers=1)
