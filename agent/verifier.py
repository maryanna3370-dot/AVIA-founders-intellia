"""Step verifier using LLM or simple heuristics."""
import os
from typing import List, Dict
try:
    import openai
except Exception:
    openai = None


def heuristic_verify(steps: List[Dict]) -> List[Dict]:
    safe = []
    for s in steps:
        text = s.get("action", "").lower()
        # reject steps that request secrets, direct infra changes, or destructive ops
        danger = ["password", "secret", "private key", "rm -rf", "sudo", "shutdown"]
        if any(d in text for d in danger):
            s["verified"] = False
            s["reason"] = "contains potentially dangerous instructions"
        else:
            s["verified"] = True
        safe.append(s)
    return safe


def llm_verify(steps: List[Dict], context: str = "") -> List[Dict]:
    if openai is None or not os.environ.get("OPENAI_API_KEY"):
        return heuristic_verify(steps)
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    try:
        prompt = "Verify the following steps for safety and provide JSON array with boolean `verified` and optional `reason` for each step. Steps: " + str([s.get("action") for s in steps])
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp["choices"][0]["message"]["content"]
        import json
        out = json.loads(text)
        # merge results
        for s, r in zip(steps, out):
            s["verified"] = bool(r.get("verified", False))
            if "reason" in r:
                s["reason"] = r.get("reason")
        return steps
    except Exception:
        return heuristic_verify(steps)


def verify_steps(steps: List[Dict], use_llm: bool = False, context: str = "") -> List[Dict]:
    if use_llm:
        return llm_verify(steps, context=context)
    return heuristic_verify(steps)
