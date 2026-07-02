"""Minimal CLI to run the Planner -> Executor loop.

Usage: python -m agent.cli --task "Write a short summary of X"
"""
import argparse
from .planner import Planner
from .executor import Executor
from .scheduler import Scheduler
from .memory import get_default_memory


def main():
    parser = argparse.ArgumentParser(description="Run minimal planner-executor agent")
    parser.add_argument("--task", required=True, help="Task prompt to decompose and execute")
    parser.add_argument("--approve", action="store_true", help="Auto-approve steps")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue task to background scheduler")
    args = parser.parse_args()

    planner = Planner()
    executor = Executor()
    memory = get_default_memory()

    plan = planner.decompose(args.task)
    print("Plan:")
    for s in plan:
        print(f" - [{s['id']}] {s['action']}")

    if args.enqueue:
        sched = Scheduler()
        task_id = sched.enqueue(args.task, approve=args.approve)
        print(f"Enqueued task {task_id}. Use memory to inspect results.")
        # For demo purposes wait briefly for the worker to run
        import time
        time.sleep(0.5)
        steps = memory.get_task_steps(task_id)
        for s in steps:
            print(f" - Step {s['step_index']}: {s['result'].get('status')}")
        sched.stop()
        return

    results = []
    for step in plan:
        if not args.approve:
            ok = input(f"Run step {step['id']}? (y/N): ")
            if ok.strip().lower() != "y":
                results.append({"step": step, "status": "skipped"})
                continue
        res = executor.execute_step(step)
        print(f"Result step {step['id']}: {res.get('status')}")
        results.append({"step": step, "result": res})

    print("\nSummary:")
    for r in results:
        step = r.get("step")
        status = r.get("result", {}).get("status", r.get("status"))
        print(f" - Step {step['id']}: {status}")


if __name__ == "__main__":
    main()
