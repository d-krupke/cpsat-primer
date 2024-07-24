from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID, uuid4


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


class TspJobRequest(BaseModel):
    tsp_instance: TspInstance = Field(..., description="The TSP instance to solve.")
    optimization_parameters: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="The optimization parameters.",
    )
    webhook_url: HttpUrl | None = Field(
        default=None, description="The URL to call once the computation is complete."
    )

    class Config:
        json_schema_extra = {
            "examples": [
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
                    "webhook_url": None,
                },
            ]
        }


class TspJobStatus(BaseModel):
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


class TspSolution(BaseModel):
    node_order: list[int] | None = Field(
        ..., description="The order of the nodes in the solution."
    )
    cost: float = Field(..., description="The cost of the solution.")
    lower_bound: float = Field(..., description="The lower bound of the solution.")
    is_infeasible: bool = Field(
        default=False, description="Whether the instance is infeasible."
    )
