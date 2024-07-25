"""
This file contains the configuration for the optimization API.
For this simple project, it only sets up the database connection and the task queue.
The other parts of the API should not be aware of the specific connection details.
"""

from db import TspJobDbConnection
import redis
from rq import Queue


def get_db_connection() -> TspJobDbConnection:
    """Provides a TspJobDbConnection instance."""
    redis_client = redis.Redis(
        host="optimization_api_redis", port=6379, decode_responses=True, db=0
    )
    return TspJobDbConnection(redis_client=redis_client)


def get_task_queue() -> Queue:
    """Provides a Redis Queue instance."""
    redis_client = redis.Redis(host="optimization_api_redis", port=6379, db=1)
    return Queue(connection=redis_client)
