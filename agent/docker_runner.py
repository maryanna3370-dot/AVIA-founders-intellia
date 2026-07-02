"""Run a tool inside Docker for stronger isolation.

This runner expects Docker to be available on the host and will mount the repo
into `/workspace` inside the container. It invokes the same `_tool_process.py`
module to execute the tool inside the container.
"""
import subprocess
import json
import os
import shlex
from typing import Dict, Any


def run_tool_in_docker(tool_module: str, args: Dict[str, Any], image: str, timeout: int = 10, resource_limits: Dict[str, Any] = None) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runner_path = "/workspace/agent/_tool_process.py"
    cmd = ["docker", "run", "--rm", "-i"]
    # resource limits: cpus, memory, user
    if resource_limits:
        cpus = resource_limits.get("cpus")
        mem = resource_limits.get("memory")
        user = resource_limits.get("user")
        if cpus:
            cmd += ["--cpus", str(cpus)]
        if mem:
            cmd += ["--memory", str(mem)]
        if user:
            cmd += ["--user", str(user)]
    # mount the repo read-only
    cmd += ["-v", f"{repo_root}:/workspace:ro"]
    # set working dir
    cmd += ["-w", "/workspace"]
    # drop capabilities for safety (best-effort)
    cmd += ["--cap-drop", "ALL"]
    cmd += [image, "python", runner_path, tool_module]
    try:
        proc = subprocess.run(cmd, input=json.dumps({"args": args}).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if proc.returncode != 0:
            # try parse stdout
            try:
                j = json.loads(proc.stdout.decode())
                if not j.get("ok"):
                    return {"ok": False, "err": j.get("err") or proc.stderr.decode()}
            except Exception:
                return {"ok": False, "err": proc.stderr.decode()}
        return json.loads(proc.stdout.decode())
    except Exception as e:
        return {"ok": False, "err": str(e)}
