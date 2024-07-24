import json
from models import TspJobStatus, TspJobRequest, TspSolution
from uuid import UUID
import redis
from typing import Optional, List
import logging


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
