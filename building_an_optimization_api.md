## Building an Optimization API

In this chapter, we will create a basic optimization service that performs
computations on a cluster rather than on the client side. This service can be
enhanced with a straightforward frontend, allowing non-technical users to access
the optimization capabilities. By encapsulating your optimization code within an
easy-to-use API, integration into larger systems becomes more streamlined, and
the separation of algorithm development from deployment is achieved.

While this chapter does not cover every aspect of building a production-ready
API, it will address many important considerations. This foundational knowledge
will enable you to collaborate effectively with your integration experts to
finalize the implementation details.

To illustrate these principles, we will develop a simple optimization model for
the Traveling Salesman Problem (TSP). Users will be able to submit an instance
of the TSP, and the API will return a solution. The primary challenge, compared
to many other APIs, is that the TSP is an NP-hard problem, and the CP-SAT solver
may need several minutes to solve even moderately sized instances. Additionally,
we cannot run the solver on the web server; instead, we must distribute the
computation across a cluster. If many requests are received simultaneously, a
request may need to wait before computation can start.

Therefore, this will not be a simple "send request, get response" API. Instead,
we will implement a task queue, returning a task ID that users can use to check
the status of their computation and retrieve the result once it is ready. To
enhance user experience, we will allow users to specify a webhook URL, which we
will call once the computation is complete.

By following this approach, we will demonstrate how to manage complex
optimization tasks within an API, providing a practical example of separating
algorithm development from deployment.

### Specifying the Essential Endpoints

Before we start coding, we should specify the endpoints our API will expose, so
we know what we need to implement.

The fundamental operations we will support are:

1. **POST /jobs**: This endpoint will accept a JSON payload containing the TSP
   instance. The API will create a new task, store the instance, and return a
   task ID. The payload will also allow users to specify a webhook URL to call
   once the computation is complete.
2. **GET /jobs/{task_id}**: This endpoint will return the status of the task
   with the given ID.
3. **GET /jobs/{task_id}/solution**: This endpoint will return the solution of
   the task with the given ID, once it is available.
4. **DELETE /jobs/{task_id}**: This endpoint will cancel the task with the given
   ID.
5. **GET /jobs**: This endpoint will return a list of all tasks, including their
   status and metadata.

By defining these endpoints, we ensure that our API is robust and capable of
handling the core functionalities required for managing and solving TSP
instances. This structure will facilitate user interactions, from submitting
tasks to retrieving solutions and monitoring the status of their requests.

Once we have successfully optimized the TSP, we can anticipate requests to
extend our optimization capabilities to other problems. Therefore, we should add
the prefix `/tsp_solver/v1` to all endpoints to facilitate future expansions of
our API with additional solvers.

Although a more hierarchical approach, such as `/solvers/tsp/v1`, could be used,
we will stick to the flat structure for simplicity in this chapter to avoid
unnecessary complexity in the code. You may ask why we do not just create a new
project for each solver and then just stick them together on a higher level. The
reason is that we may want to share the same infrastructure for all solvers,
especially the task queue and the worker cluster. Therefore, it makes sense to
keep them in the same project. However, it will make sense to separate the
actual algorithms from the API code, and only import the algorithms into our API
project. We will not do this in this chapter, but I personally prefer the
algorithms to be as separated as possible as they are often complex enough on
their own.

## Overview

This project aims to develop a scalable and efficient web API for solving the
Traveling Salesman Problem (TSP) using modern technologies. The API will
leverage FastAPI for handling web requests, Redis for data storage, and RQ for
task queue management. The goal is to create an API that accepts TSP instances,
processes them asynchronously, and returns solutions, facilitating easy
integration into larger systems.

### Key Components

1. **FastAPI**: FastAPI is a modern, fast (high-performance) web framework for
   building APIs with Python 3.6+ based on standard Python type hints. We use
   FastAPI to define API endpoints and handle HTTP requests due to its
   simplicity, speed, and automatic interactive API documentation.

2. **Redis**: Redis is an in-memory data structure store that can be used as a
   database, cache, and message broker. We use Redis for its speed and
   efficiency in storing tasks and solutions, allowing quick access and
   automatic expiration of data when it is no longer needed.

3. **RQ (Redis Queue)**: RQ is a simple Python library for queuing jobs and
   processing them in the background with workers. This allows our API to handle
   tasks asynchronously, offloading computationally expensive processes to
   background workers and thus improving the API's responsiveness.

4. **OR-Tools**: OR-Tools is an open-source software suite for optimization,
   developed by Google. It provides various algorithms for solving optimization
   problems, including the TSP. We use OR-Tools for its robust and efficient
   implementations.

### Project Structure

1. **Requirements**: We define the necessary Python packages in a
   `requirements.txt` file to ensure that the environment can be easily set up
   and replicated. This file includes all dependencies needed for the project.

2. **Docker Environment**:

   - **Dockerfile**: The Dockerfile specifies the Docker image and environment
     setup for the API. It ensures that the application runs in a consistent
     environment across different machines.
   - **docker-compose.yml**: This file configures the services required for the
     project, including the API, Redis, and worker instances. Docker Compose
     simplifies the process of managing multiple containers, ensuring they are
     correctly built and started in the right order.

3. **Solver Implementation**:

   - `solver.py`: This script contains the implementation of the TSP solver
     using OR-Tools. It defines how the TSP instance is solved and how the
     results are structured. The solver takes in a set of locations and returns
     the optimal route.

4. **Request and Response Models**:

   - `models.py`: This module defines data models for API requests and responses
     using Pydantic. Pydantic ensures data validation and serialization, which
     helps maintain the integrity and consistency of data being processed by the
     API.

5. **Database**:

   - `db.py`: This script implements a proxy class to interact with Redis,
     abstracting database operations for storing and retrieving job requests,
     statuses, and solutions. It provides a clean interface for the rest of the
     application to interact with Redis without needing to handle low-level
     details.

6. **Config**:

   - `config.py`: This module provides configuration functions to set up the
     database connection and task queue. By centralizing configuration, we
     ensure that other parts of the application do not need to manage connection
     details, making the codebase more modular and easier to maintain.

7. **Tasks**:

   - `tasks.py`: This script defines the tasks that fetch job data, run the
     optimization algorithm, store results, and send notifications via webhooks.
     These tasks will be outsourced to separate workers, and the API just needs
     to have a reference to the task functions in order to queue them. For the
     workers, this file will be the entry point.

8. **API**:
   - `main.py`: This script implements the FastAPI application with routes for
     submitting jobs, checking job statuses, retrieving solutions, and canceling
     jobs. It integrates all components to provide a functional API that users
     can interact with.

### Running the Application

To run the application, use Docker and Docker Compose to build and run the
containers. This ensures the API and its dependencies are correctly set up. Once
the containers are running, you can interact with the API via HTTP requests to
submit TSP instances, monitor job progress, retrieve solutions, and manage
tasks.

By the end of this project, you will have a robust API capable of solving TSP
instances asynchronously, providing a foundation for deploying optimization
algorithms as web services.

## Requirements

We will use FastAPI as the web framework for our API because it is modern,
efficient, and easy to use. Redis will serve as our database, and RQ will manage
the task queue. These technologies are chosen for their performance, ease of
use, and community support.

### requirements.txt

```plaintext
fastapi
ortools
redis
rq
```

## Docker Environment

We use Docker to ensure a consistent development and production environment.
Docker allows us to package our application with all its dependencies into a
standardized unit for software development. Docker Compose is used to manage
multi-container applications, defining and running multi-container Docker
applications.

### Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the current directory contents into the container at /app
COPY ./app /app
```

The Dockerfile defines the environment setup for the API. It starts with a slim
version of Python 3.12, sets the working directory, installs required packages,
and copies the application code into the container.

### docker-compose.yml

```yaml
services:
  optimization_api_fastapi:
    build: .
    container_name: optimization_api_fastapi
    ports:
      - "80:80"
    depends_on:
      - optimization_api_redis
    command: python3 -m uvicorn main:app --host 0.0.0.0 --port 80 --reload

  optimization_api_redis:
    image: redis:latest
    container_name: optimization_api_redis

  optimization_api_worker:
    build: .
    command:
      rq worker --with-scheduler --url redis://optimization_api_redis:6379/1
    depends_on:
      - optimization_api_redis
    deploy:
      replicas: 2 # Adding two workers for parallel processing
```

The `docker-compose.yml` file sets up three services:

- `optimization_api_fastapi`: This service builds the FastAPI application,
  exposes it on port 80, and ensures it starts after Redis is available.
- `optimization_api_redis`: This service uses the latest Redis image to provide
  in-memory data storage.
- `optimization_api_worker`: This service builds the worker, which processes
  tasks from the queue. We can scale the number of workers by increasing the
  number of replicas. Theoretically, these workers could be run on different
  machines to scale horizontally.

This setup ensures that all components are correctly orchestrated, enabling a
seamless development and deployment experience.

### Solver

In this section, we will explore the implementation of the optimization
algorithm that we will deploy as an API. Specifically, we will focus on a simple
implementation of the Traveling Salesman Problem (TSP) using the `add_circuit`
constraint from the CP-SAT solver in OR-Tools.

The solver is the core component of our application, responsible for finding the
optimal solution to the TSP instance provided by the user. The algorithm is
implemented directly in the API project for simplicity. However, for more
complex optimization algorithms, it is advisable to separate the algorithm into
a distinct module or project. This separation facilitates isolated testing and
benchmarking of the algorithm and improves the development process, especially
when working in a team where different teams might maintain the API and the
optimization algorithm.

```python
# ./app/solver.py
from typing import Callable
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# A precise definition of the input and output data for the TSP solver.
# ---------------------------------------------------------------------


class DirectedEdge(BaseModel):
    source: int = Field(..., ge=0, description="The source node of the edge.")
    target: int = Field(..., ge=0, description="The target node of the edge.")
    cost: int = Field(..., ge=0, description="The cost of traversing the edge.")


class TspInstance(BaseModel):
    num_nodes: int = Field(
        ..., gt=0, description="The number of nodes in the TSP instance."
    )
    edges: list[DirectedEdge] = Field(
        ..., description="The directed edges of the TSP instance."
    )


class OptimizationParameters(BaseModel):
    timeout: int = Field(
        default=60,
        gt=0,
        description="The maximum time in seconds to run the optimization.",
    )


class TspSolution(BaseModel):
    node_order: list[int] | None = Field(
        ..., description="The order of the nodes in the solution."
    )
    cost: float = Field(..., description="The cost of the solution.")
    lower_bound: float = Field(..., description="The lower bound of the solution.")
    is_infeasible: bool = Field(
        default=False, description="Whether the instance is infeasible."
    )


# ---------------------------------------------------------------------
# The TSP solver implementation using the CP-SAT solver from OR-Tools.
# ---------------------------------------------------------------------


class TspSolver:
    def __init__(
        self, tsp_instance: TspInstance, optimization_parameters: OptimizationParameters
    ):
        self.tsp_instance = tsp_instance
        self.optimization_parameters = optimization_parameters
        self.model = cp_model.CpModel()
        self.edge_vars = {
            (edge.source, edge.target): self.model.new_bool_var(
                f"x_{edge.source}_{edge.target}"
            )
            for edge in tsp_instance.edges
        }
        self.model.minimize(
            sum(
                edge.cost * self.edge_vars[(edge.source, edge.target)]
                for edge in tsp_instance.edges
            )
        )
        self.model.add_circuit(
            [(source, target, var) for (source, target), var in self.edge_vars.items()]
        )

    def solve(self, log_callback: Callable[[str], None] | None = None):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.optimization_parameters.timeout
        if log_callback:
            solver.parameters.log_search_progress = True
            solver.log_callback = log_callback
        status = solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return TspSolution(
                node_order=[
                    source
                    for (source, target), var in self.edge_vars.items()
                    if solver.value(var)
                ],
                cost=solver.objective_value,
                lower_bound=solver.best_objective_bound,
            )
        if status == cp_model.INFEASIBLE:
            return TspSolution(
                node_order=None,
                cost=float("inf"),
                lower_bound=float("inf"),
                is_infeasible=True,
            )
        return TspSolution(
            node_order=None,
            cost=float("inf"),
            lower_bound=solver.best_objective_bound,
        )
```

> [!TIP]
>
> CP-SAT itself uses Protobuf for its input, output, and configuration. Having
> well-defined data models can help prevent many "garbage in, garbage out"
> issues and ease integration with other systems. It also facilitates testing
> and debugging, as you can simply serialize a specific scenario. For
> configuration, having default values is very helpful, as it allows you to
> extend the configuration without breaking backward compatibility. This can be
> a significant advantage, as you usually do not know all requirements upfront.
> Pydantic performs this job very well and can be used for the web API as well.
> Protobuf, while not Python-specific and therefore more versatile, is more
> complex to use and lacks the same flexibility as Pydantic.

## Request and Response Models

In this section, we will define the request and response models for the API.
These models will facilitate the communication between the client and the server
by ensuring that the data exchanged is structured and validated correctly. We
use Pydantic to define these models, leveraging its powerful data validation and
serialization capabilities.

### Implementation of Request and Response Models

The models are defined in the `models.py` file and include the necessary data
structures for submitting a TSP job request and tracking the status of the job.

```python
# ./app/models.py
"""
This file contains the implementation of additional data models for the optimization API.
"""

from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID, uuid4
from solver import OptimizationParameters, TspInstance
```

### TSP Job Request Model

The `TspJobRequest` model encapsulates the information required to submit a TSP
job to the API. It includes the TSP instance, optimization parameters, and an
optional webhook URL for notifications upon job completion.

```python
class TspJobRequest(BaseModel):
    """
    A request model for a TSP job.
    """

    tsp_instance: TspInstance = Field(..., description="The TSP instance to solve.")
    optimization_parameters: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="The optimization parameters.",
    )
    webhook_url: HttpUrl | None = Field(
        default=None, description="The URL to call once the computation is complete."
    )
```

An request could look as follows:

```json
{
    "tsp_instance": {
        "num_nodes": 4,
        "edges": [
            {"source": 0, "target": 1, "cost": 1},
            {"source": 1, "target": 2, "cost": 2},
            {"source": 2, "target": 3, "cost": 3},
            {"source": 3, "target": 0, "cost": 4},
        ],
    },
    "optimization_parameters": {"timeout": 5},
    "webhook_url": null,
},
```

### TSP Job Status Model

The `TspJobStatus` model is used to track the status of a TSP job. It provides
fields to monitor various stages of the job lifecycle, from submission to
completion.

```python
class TspJobStatus(BaseModel):
    """
    A response model for the status of a TSP job.
    """

    task_id: UUID = Field(default_factory=uuid4, description="The ID of the task.")
    status: str = Field(default="Submitted", description="The status of the task.")
    submitted_at: datetime = Field(
        default_factory=datetime.now, description="The time the task was submitted."
    )
    started_at: datetime | None = Field(
        default=None, description="The time the task was started."
    )
    completed_at: datetime | None = Field(
        default=None, description="The time the task was completed."
    )
    error: str | None = Field(
        default=None, description="The error message if the task failed."
    )
```

These models ensure that the data exchanged between the client and the server is
well-defined and validated, providing a robust framework for handling TSP job
requests and tracking their status.

## Database

In this section, we will implement a database to store the tasks and solutions.
For simplicity, we use Redis, which serves as both our database and task queue.
This approach minimizes the need to set up additional databases and leverages
Redis's key-value storage and automatic data expiration features. To ensure
flexibility, we wrap Redis operations in a proxy class, allowing us to switch
the database backend easily if needed.

### Implementation of the Database Proxy Class

The `TspJobDbConnection` class encapsulates the interactions with the Redis
database. It provides methods to register new jobs, update job statuses,
retrieve job requests, statuses, and solutions, list all jobs, and delete jobs.

```python
# ./app/db.py
"""
This file contains a proxy class to interact with the database.
We are using Redis as the database for this example, but the implementation
can be easily adapted to other databases, as the proxy class abstracts the
database operations.
"""

import json
from models import TspJobStatus, TspJobRequest
from solver import TspSolution
from uuid import UUID
import redis
from typing import Optional, List
import logging
```

### Initialization and Helper Methods

The class is initialized with a Redis client and an expiration time for the
stored data. The `_get_data` method is a helper that retrieves and parses JSON
data from Redis by key.

```python
class TspJobDbConnection:
    def __init__(self, redis_client: redis.Redis, expire_time: int = 24 * 60 * 60):
        """Initialize the Redis connection and expiration time."""
        self._redis = redis_client
        self._expire_time = expire_time
        logging.basicConfig(level=logging.INFO)

    def _get_data(self, key: str) -> Optional[dict]:
        """Get data from Redis by key and parse JSON."""
        try:
            data = self._redis.get(key)
            if data is not None:
                return json.loads(data)
        except redis.RedisError as e:
            logging.error(f"Redis error: {e}")
        return None
```

### Retrieve Data Methods

The `get_request`, `get_status`, and `get_solution` methods retrieve a TSP job
request, status, and solution, respectively, by their task ID.

```python
def get_request(self, task_id: UUID) -> Optional[TspJobRequest]:
    """Retrieve a TSP job request by task ID."""
    data = self._get_data(f"request:{task_id}")
    return TspJobRequest(**data) if data else None


def get_status(self, task_id: UUID) -> Optional[TspJobStatus]:
    """Retrieve a TSP job status by task ID."""
    data = self._get_data(f"status:{task_id}")
    return TspJobStatus(**data) if data else None


def get_solution(self, task_id: UUID) -> Optional[TspSolution]:
    """Retrieve a TSP solution by task ID."""
    data = self._get_data(f"solution:{task_id}")
    return TspSolution(**data) if data else None
```

### Store Data Methods

The `set_solution` method stores a TSP solution in Redis with an expiration
time. The `register_job` method registers a new TSP job request and status in
Redis.

```python
def set_solution(self, task_id: UUID, solution: TspSolution) -> None:
    """Set a TSP solution in Redis with an expiration time."""
    try:
        self._redis.set(
            f"solution:{task_id}", solution.model_dump_json(), ex=self._expire_time
        )
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)


def register_job(self, request: TspJobRequest) -> TspJobStatus:
    """Register a new TSP job request and status in Redis."""
    job_status = TspJobStatus()
    try:
        pipeline = self._redis.pipeline()
        pipeline.set(
            f"status:{job_status.task_id}",
            job_status.model_dump_json(),
            ex=self._expire_time,
        )
        pipeline.set(
            f"request:{job_status.task_id}",
            request.model_dump_json(),
            ex=self._expire_time,
        )
        pipeline.execute()
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)

    return job_status
```

### Update and List Jobs Methods

The `update_job_status` method updates the status of an existing TSP job. The
`list_jobs` method lists all TSP job statuses.

```python
def update_job_status(self, job_status: TspJobStatus) -> None:
    """Update the status of an existing TSP job."""
    try:
        self._redis.set(
            f"status:{job_status.task_id}",
            job_status.model_dump_json(),
            ex=self._expire_time,
        )
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)


def list_jobs(self) -> List[TspJobStatus]:
    """List all TSP job statuses."""
    try:
        status_keys = self._redis.keys("status:*")
        data = self._redis.mget(status_keys)
        return [TspJobStatus(**json.loads(status)) for status in data if status]
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)

        return []
```

### Delete Job Method

The `delete_job` method deletes a TSP job request, status, and solution from
Redis.

```python
def delete_job(self, task_id: UUID) -> None:
    """Delete a TSP job request, status, and solution from Redis."""
    try:
        pipeline = self._redis.pipeline()
        pipeline.delete(f"status:{task_id}")
        pipeline.delete(f"request:{task_id}")
        pipeline.delete(f"solution:{task_id}")
        pipeline.execute()
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)
```

## Config

In this section, we will set up the configuration for the optimization API. The
configuration will ensure that the database and the task queue are properly set
up, allowing the rest of the code to remain unaware of the specific connection
details. This separation of concerns enhances maintainability and modularity.

### Implementation of the Configuration File

The `config.py` file contains functions to set up and provide the database
connection and task queue. This centralized configuration allows the rest of the
application to use these resources without needing to handle their
initialization or connection details.

```python
# ./app/config.py
"""
This file contains the configuration for the optimization API.
For this simple project, it only sets up the database connection and the task queue.
The other parts of the API should not be aware of the specific connection details.
"""

from db import TspJobDbConnection
import redis
from rq import Queue
```

### Database Connection Function

The `get_db_connection` function sets up the Redis client for the database and
returns an instance of `TspJobDbConnection`.

```python
def get_db_connection() -> TspJobDbConnection:
    """Provides a TspJobDbConnection instance."""
    redis_client = redis.Redis(
        host="optimization_api_redis", port=6379, decode_responses=True, db=0
    )
    return TspJobDbConnection(redis_client=redis_client)
```

### Task Queue Function

The `get_task_queue` function sets up the Redis client for the task queue and
returns an instance of `Queue`.

```python
def get_task_queue() -> Queue:
    """Provides a Redis Queue instance."""
    redis_client = redis.Redis(host="optimization_api_redis", port=6379, db=1)
    return Queue(connection=redis_client)
```

## Tasks

With the database in place, we can create the tasks that will run the
optimization. The optimization will run in a separate process and use the
database to communicate with the web server. To keep things simple, we will pass
only the job reference to the task. The task will fetch the necessary data from
the database and update the database with the results. Additionally, by
including an `if __name__ == "__main__":` block, we allow the tasks to be run
via an external task queue as system commands.

### Implementation of the Tasks

The `tasks.py` file contains functions and logic for running the optimization
job in a separate worker process.

```python
# ./app/tasks.py
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
```

### Webhook Notification Function

The `send_webhook` function sends a POST request to the specified webhook URL
with the job status. This allows for asynchronous notifications when the
computation is complete.

```python
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
```

### Optimization Job Function

The `run_optimization_job` function fetches the job request from the database,
runs the optimization algorithm, and stores the solution back in the database.
It also updates the job status and sends a webhook notification upon completion.

```python
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
```

## API

In this final section, we will build the actual API using FastAPI. This API will
expose endpoints to submit TSP job requests, check job statuses, retrieve
solutions, cancel jobs, and list all jobs. FastAPI provides an efficient and
easy-to-use framework for building web APIs with Python.

### Implementation of the API

The `main.py` file contains the FastAPI application setup and the API routes.
For simplicity, all routes are included in a single file, but in larger
projects, it is advisable to separate them into different modules.

```python
# ./app/main.py
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
```

### FastAPI Application Setup

The FastAPI application is initialized with a title and description. An API
router is created to group the routes related to the TSP solver.

```python
app = FastAPI(
    title="My Optimization API",
    description="This is an example on how to deploy an optimization algorithm based on CP-SAT as an API.",
)

tsp_solver_v1_router = APIRouter(tags=["TSP_solver_v1"])
```

### API Routes

#### Submit a New Job

The `post_job` endpoint allows users to submit a new TSP job. The job is
registered in the database, and the optimization task is enqueued in the task
queue for asynchronous processing.

```python
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
    # enqueue the optimization job in the task queue.
    # Will return immediately, the job will be run in a separate worker.
    task_queue.enqueue(
        run_optimization_job,
        job_status.task_id,
        # adding a 60 second buffer to the job timeout
        job_timeout=job_request.optimization_parameters.timeout + 60,
    )
    return job_status
```

#### Get Job Status

The `get_job` endpoint returns the status of a specific job identified by its
task ID.

```python
@tsp_solver_v1_router.get("/jobs/{task_id}", response_model=TspJobStatus)
def get_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the status of a job.
    """
    status = db_connection.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
```

#### Get Job Solution

The `get_solution` endpoint returns the solution of a specific job if it is
available.

```python
@tsp_solver_v1_router.get("/jobs/{task_id}/solution", response_model=TspSolution)
def get_solution(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the solution of a job, if available.
    """
    solution = db_connection.get_solution(task_id)
    if solution is None:
        raise HTTPException(status_code=404, detail="Solution not found")
    return solution
```

#### Cancel a Job

The `cancel_job` endpoint deletes or cancels a job. It does not immediately stop
the job if it is already running.

```python
@tsp_solver_v1_router.delete("/jobs/{task_id}")
def cancel_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Deletes/cancels a job. This will *not* immediately stop the job if it is running.
    """
    db_connection.delete_job(task_id)
```

#### List All Jobs

The `list_jobs` endpoint returns a list of all jobs and their statuses.

```python
@tsp_solver_v1_router.get("/jobs", response_model=list[TspJobStatus])
def list_jobs(db_connection=Depends(get_db_connection)):
    """
    List all jobs.
    """
    return db_connection.list_jobs()
```

### Including the Router and Running the Application

The router is included in the FastAPI application with a specific prefix, and
the application is set up to run with Uvicorn when executed as a script.

```python
app.include_router(tsp_solver_v1_router, prefix="/tsp_solver/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, workers=1)
```
