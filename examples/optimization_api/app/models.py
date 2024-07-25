"""
This file contains the implementation of additional data models for the optimization API.
"""

from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID, uuid4

from solver import OptimizationParameters, TspInstance


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
