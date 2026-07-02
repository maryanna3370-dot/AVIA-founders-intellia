"""Executor that calls tool adapters in `tools` package with retries and routing.

Tools must expose `run(args)` and return JSON-serializable dicts. Executor supports
per-step override args: `max_retries`, `timeout_s`, and will attach metadata.
"""
from typing import Any, Dict
import importlib
import time
import traceback
import subprocess
import json
import shlex
import sys
import os
from agent import config


class Executor:
    def __init__(self, tools_package: str = "tools", default_retries: int = 2):
        self.tools_package = tools_package
        self.default_retries = default_retries

    def _import_tool(self, tool_name: str):
        try:
            return importlib.import_module(f"{self.tools_package}.{tool_name}")
        except Exception:
            return None

    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool")
        args = step.get("args", {}) or {}
        max_retries = int(args.get("max_retries", step.get("max_retries", self.default_retries)))
        backoff = float(args.get("backoff", 0.5))

        # Authorization: ensure tool is allowed
        if config.ALLOWED_TOOLS is not None and tool_name not in config.ALLOWED_TOOLS:
            return {"status": "error", "error": "tool_not_allowed"}

        mod = self._import_tool(tool_name)
        if mod is None:
            return {"status": "error", "error": f"import_error: {tool_name}"}

        if not hasattr(mod, "run"):
            return {"status": "error", "error": "tool_missing_run"}

        attempt = 0
        last_exc = None
        start_time = time.time()
        while attempt <= max_retries:
            attempt += 1
            try:
                # Optionally sandbox tool execution in a separate process using a runner
                if config.SANDBOX_TOOLS:
                    runner = os.path.join(os.path.dirname(__file__), "_tool_process.py")
                    cmd = [sys.executable, runner, f"{self.tools_package}.{tool_name}"]
                    if config.SANDBOX_MODE == "docker":
                        # use Docker runner
                        from .docker_runner import run_tool_in_docker
                        res = run_tool_in_docker(f"{self.tools_package}.{tool_name}", args, config.DOCKER_IMAGE, timeout=int(args.get("timeout", 10)))
                        if not res.get("ok"):
                            raise Exception(res.get("err"))
                        out = res.get("out")
                    else:
                        env = os.environ.copy()
                        repo_root = os.path.dirname(os.path.dirname(__file__))
                        prev = env.get("PYTHONPATH", "")
                        env["PYTHONPATH"] = repo_root + (":" + prev if prev else "")
                        proc = subprocess.run(
                            cmd,
                            input=json.dumps({"args": args}).encode(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=args.get("timeout", 10),
                            env=env,
                        )
                        if proc.returncode != 0:
                            # try to parse stdout for error
                            try:
                                j = json.loads(proc.stdout.decode(errors="ignore"))
                                if not j.get("ok"):
                                    raise Exception(j.get("err") or "tool_error")
                            except Exception:
                                raise Exception(proc.stderr.decode(errors="ignore") or "tool_error")
                        j = json.loads(proc.stdout.decode())
                        if not j.get("ok"):
                            raise Exception(j.get("err"))
                        out = j.get("out")
                    if proc.returncode != 0:
                        # try to parse stdout for error
                        try:
                            j = json.loads(proc.stdout.decode(errors="ignore"))
                            if not j.get("ok"):
                                raise Exception(j.get("err") or "tool_error")
                        except Exception:
                            raise Exception(proc.stderr.decode(errors="ignore") or "tool_error")
                    j = json.loads(proc.stdout.decode())
                    if not j.get("ok"):
                        raise Exception(j.get("err"))
                    out = j.get("out")
                else:
                    out = mod.run(args)

                elapsed = time.time() - start_time
                return {
                    "status": "ok",
                    "output": out,
                    "meta": {"attempts": attempt, "elapsed_s": elapsed},
                }
            except Exception as e:
                last_exc = e
                if attempt > max_retries:
                    tb = traceback.format_exc()
                    return {"status": "error", "error": str(e), "trace": tb}
                # exponential backoff
                time.sleep(backoff * (2 ** (attempt - 1)))


if __name__ == "__main__":
    from agent.planner import Planner
    p = Planner()
    steps = p.decompose("Quickly fetch intro about product-market fit.")
    exe = Executor()
    for s in steps:
        print(exe.execute_step(s))
