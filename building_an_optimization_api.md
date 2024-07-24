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

Before we start coding, we should specify the endpoints our API will expose.

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

#### Endpoint Details

An endpoint requires more specification than just a name and an operator. We
need at least to define a request and response format (error messages can mainly
follow the common conventions, though, we should be more specific for actual
production use).

I really like to use FastAPI for building APIs, as it will handle the API
documentation for use via pydantic models. Instead of having to write JSON
schemas, we can completely stay within Python and just specify the pydantic
models we want to use. If you are new to pydantic, it may be worth reading a
quick introduction to it. It is a very powerful library that can be used for
many things beyond just API definitions. However, pydantic can also be learned
by example, so we will just use it here.

Let us start with the request and response models for the `/jobs` endpoint. We
will use the following models:

```python
from pydantic import BaseModel, HttpUrl, Field


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


class JobRequest(BaseModel):
    tsp_instance: TspInstance = Field(..., description="The TSP instance to solve.")
    optimization_parameters: OptimizationParameters = Field(
        ..., description="The optimization parameters."
    )
    webhook_url: HttpUrl = Field(
        None, description="The URL to call once the computation is complete."
    )
```

While we could just return the task ID, we may want to extend it by further
information, such as the priority of the task. We make the task ID a string so
we can use UUIDs, which cannot be easily guessed in case we do not want the job
id to give any information about the internal state of the system.

```python
class JobResponse(BaseModel):
    task_id: str = Field(..., description="The ID of the task.")
    submitted_at: datetime = Field(..., description="The time the task was submitted.")
```

The other endpoints do not require a request model, but we will need a response
model for some of them. We will use the following models:

```python
class JobStatus(BaseModel):
    task_id: str = Field(..., description="The ID of the task.")
    status: str = Field(..., description="The status of the task.")
    submitted_at: datetime = Field(..., description="The time the task was submitted.")
    started_at: datetime | None = Field(
        None, description="The time the task was started."
    )
    completed_at: datetime | None = Field(
        None, description="The time the task was completed."
    )
    error: str | None = Field(None, description="The error message if the task failed.")


class TspSolution(BaseModel):
    node_order: List[int] = Field(
        ..., description="The order of the nodes in the solution."
    )
    cost: int = Field(..., description="The cost of the solution.")
    lower_bound: int = Field(..., description="The lower bound of the solution.")
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

### Implementing the API

Now that we have defined the endpoints and models, we can start implementing the
API. We will use FastAPI, as it is a modern and efficient web framework for
building APIs with Python. FastAPI is based on standard Python type hints, which
makes it easy to use and understand. It also provides automatic generation of
OpenAPI documentation, which is a significant advantage for API development.

We will skip the actual execution of the optimization for now and just return a
dummy solution, so we can focus on the API implementation first.
