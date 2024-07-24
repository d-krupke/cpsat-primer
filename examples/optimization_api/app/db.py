import json
from models import TspJobStatus, TspJobRequest, TspSolution
from uuid import UUID
import redis
from rq import Queue


class TspJobDbConnection:
    def __init__(self):
        self._redis = redis.Redis(
            host="optimization_api_redis", port=6379, decode_responses=True, db=0
        )
        self._expire_time = 24 * 60 * 60

    def get_request(self, task_id: UUID) -> TspJobRequest | None:
        data = self._redis.get(f"request:{task_id}")
        if data is None:
            return None
        parsed_data = json.loads(data)
        return TspJobRequest(**parsed_data)

    def get_status(self, task_id: UUID) -> TspJobStatus | None:
        data = self._redis.get(f"status:{task_id}")
        if data is None:
            return None
        parsed_data = json.loads(data)
        return TspJobStatus(**parsed_data)

    def get_solution(self, task_id: UUID) -> TspSolution | None:
        data = self._redis.get(f"solution:{task_id}")
        if data is None:
            return None
        parsed_data = json.loads(data)
        return TspSolution(**parsed_data)

    def set_solution(self, task_id: UUID, solution: TspSolution):
        self._redis.set(
            f"solution:{task_id}", solution.model_dump_json(), ex=self._expire_time
        )

    def register_job(self, request: TspJobRequest) -> TspJobStatus:
        job_status = TspJobStatus()
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
        return job_status

    def update_job_status(self, job_status: TspJobStatus):
        self._redis.set(
            f"status:{job_status.task_id}",
            job_status.model_dump_json(),
            ex=self._expire_time,
        )

    def list_jobs(self) -> list[TspJobStatus]:
        status_keys = list(self._redis.keys("status:*"))
        data = self._redis.mget(status_keys)
        all_status = []
        for status in data:
            if status is None:
                continue
            parsed_data = json.loads(status)
            all_status.append(TspJobStatus(**parsed_data))
        return all_status

    def delete_job(self, task_id: UUID):
        self._redis.delete(f"status:{task_id}")
        self._redis.delete(f"request:{task_id}")
        self._redis.delete(f"solution:{task_id}")


db_connection = TspJobDbConnection()
task_queue = Queue(
    connection=redis.Redis(host="optimization_api_redis", port=6379, db=1)
)
