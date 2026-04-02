"""
Celery application for long-running background tasks.

Requires Redis: brew install redis && redis-server
  or: docker run -d -p 6379:6379 redis:alpine

Start worker:
  cd src && celery -A web.backend.celery_app worker --loglevel=info
"""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "signal_engine",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # 1 hour
    worker_prefetch_multiplier=1,  # One task at a time (CPU-bound)
    worker_concurrency=1,  # Single worker for heavy computation
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
)

# Auto-discover tasks module
celery_app.autodiscover_tasks(["web.backend"])
