"""Simple planner that decomposes a task into sequential steps.

This is intentionally minimal for a runnable prototype demonstrating
planner->executor interaction.
"""
from typing import List, Dict
from . import config
from . import verifier


class Planner:
    def __init__(self, llm_planner=None):
        self.llm_planner = llm_planner

    def decompose(self, task: str) -> List[Dict]:
        """Return a list of step dicts for the given task.

        Uses LLM planner when configured; otherwise falls back to simple heuristic.
        """
        if config.USE_LLM and self.llm_planner is not None:
            try:
                steps = self.llm_planner.decompose(task)
                # verify steps
                steps = verifier.verify_steps(steps, use_llm=True, context=task)
                return steps
            except Exception:
                pass

        raw_steps = [s.strip() for s in task.replace('?', '.').split('.') if s.strip()]
        steps = []
        for i, s in enumerate(raw_steps, start=1):
            steps.append({
                "id": i,
                "action": s,
                "tool": "sample_tool",
                "args": {"query": s},
            })
        # perform heuristic verification even when not using LLM planner
        steps = verifier.verify_steps(steps, use_llm=False, context=task)
        return steps


if __name__ == "__main__":
    from .llm import get_llm_planner
    p = Planner(llm_planner=get_llm_planner())
    print(p.decompose("Summarize market sizing. Find competitors."))
