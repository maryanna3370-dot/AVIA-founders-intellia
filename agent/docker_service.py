"""A small Docker-runner service that listens for jobs on Redis and runs them.

Job format (pushed to Redis list `docker_jobs` as JSON string):
{ "job_id": "<id>", "tool_module": "tools.sample_tool", "args": {...}, "image": "optional" }

Result is stored in Redis key `docker_results:{job_id}` as JSON string and a pubsub
message is published on channel `docker_done:{job_id}`.

This is a minimal prototype; for production use an orchestrator and proper security.
"""
import os
import json
import time
from typing import Dict, Any

try:
    import redis
except Exception:
    redis = None

from .docker_runner import run_tool_in_docker
from .config import DOCKER_IMAGE


def run_service_once(redis_url: str, listen_list: str = "docker_jobs") -> None:
    if redis is None:
        raise RuntimeError("redis package required for docker_service")
    r = redis.from_url(redis_url)
    print("Docker service listening on Redis list", listen_list)
    while True:
        item = r.brpop(listen_list, timeout=5)
        if not item:
            continue
        _, payload = item
        try:
            job = json.loads(payload)
            job_id = job.get("job_id") or str(int(time.time() * 1000))
            tool_module = job.get("tool_module")
            args = job.get("args", {})
            image = job.get("image", DOCKER_IMAGE)
            timeout = int(job.get("timeout", 10))
            res = run_tool_in_docker(tool_module, args, image, timeout=timeout)
            key = f"docker_results:{job_id}"
            r.set(key, json.dumps(res))
            r.publish(f"docker_done:{job_id}", json.dumps({"job_id": job_id}))
        except Exception as e:
            print("Job failed:", e)


def main():
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("Set REDIS_URL to use docker_service")
        return
    run_service_once(redis_url)


if __name__ == "__main__":
    main()
