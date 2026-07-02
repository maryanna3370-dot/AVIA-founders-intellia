"""Scheduler with prioritization, delayed enqueue, per-step approvals, and workers."""
import threading
import queue
import uuid
import time
from typing import Any, Dict
from .planner import Planner
from .executor import Executor
from .memory import get_default_memory


class PrioritizedItem:
    def __init__(self, priority: int, enqueue_time: float, job: Dict[str, Any]):
        self.priority = priority
        self.enqueue_time = enqueue_time
        self.job = job

    def __lt__(self, other):
        # lower priority number means higher priority
        if self.priority == other.priority:
            return self.enqueue_time < other.enqueue_time
        return self.priority < other.priority


class Scheduler:
    def __init__(self, worker_count: int = 1):
        self.q = queue.PriorityQueue()
        self.workers = []
        self.worker_count = max(1, worker_count)
        self._stop = threading.Event()
        self.planner = Planner()
        self.executor = Executor()
        self.memory = get_default_memory()
        for _ in range(self.worker_count):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def enqueue(self, prompt: str, task_id: str = None, approve: bool = False, priority: int = 100, delay_s: float = 0.0) -> str:
        task_id = task_id or str(uuid.uuid4())
        self.memory.save_task(task_id, prompt, {"approve": approve})
        plan = self.planner.decompose(prompt)
        now = time.time()
        for step in plan:
            step_index = step.get("id", 0)
            # save as pending step
            self.memory.save_step_pending(task_id, step_index, step)
            job = {"task_id": task_id, "step": step, "approve": approve}
            # compute effective enqueue time including per-step delay
            enq_time = now + float(step.get("delay_s", delay_s))
            item = PrioritizedItem(priority, enq_time, job)
            self.q.put(item)
        return task_id

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except Exception:
                continue

            # respect delayed enqueue time
            if hasattr(item, "enqueue_time") and item.enqueue_time > time.time():
                # not ready yet, put back and sleep briefly
                self.q.put(item)
                time.sleep(0.2)
                continue

            job = item.job
            task_id = job["task_id"]
            step = job.get("step")
            approve = job.get("approve", False)
            step_index = step.get("id", 0)

            # if approval required, wait for approval (use event-driven wait if supported)
            if not approve:
                wait_fn = getattr(self.memory, "wait_for_approval", None)
                if callable(wait_fn):
                    ok = wait_fn(task_id, step_index, timeout=30.0)
                    if not ok:
                        self.memory.save_step_result(task_id, step_index, step, {"status": "skipped", "reason": "no_approval"})
                        self.q.task_done()
                        continue
                else:
                    waited = 0.0
                    while waited < 30.0:
                        pending = self.memory.get_pending_steps()
                        matched = [p for p in pending if p["task_id"] == task_id and p["step_index"] == step_index]
                        if not matched:
                            break
                        time.sleep(0.5)
                        waited += 0.5
                    if waited >= 30.0:
                        self.memory.save_step_result(task_id, step_index, step, {"status": "skipped", "reason": "no_approval"})
                        self.q.task_done()
                        continue

            res = self.executor.execute_step(step)
            self.memory.save_step_result(task_id, step_index, step, res)
            self.q.task_done()

    def stop(self):
        self._stop.set()
        # join workers briefly
        for w in self.workers:
            w.join(timeout=1)


if __name__ == "__main__":
    s = Scheduler()
    tid = s.enqueue("Find competitors and summarize market sizing.")
    time.sleep(1)
    print(s.memory.get_task_steps(tid))
