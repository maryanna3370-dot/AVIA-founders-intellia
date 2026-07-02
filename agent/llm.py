"""LLM planner interface with optional OpenAI implementation and a deterministic fallback."""
import os
from typing import List, Dict
try:
    import openai
except Exception:
    openai = None
try:
    from jsonschema import validate as jsonschema_validate
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
except Exception:
    jsonschema_validate = None
    JsonSchemaValidationError = Exception
from .schema import STEPS_ARRAY_SCHEMA


class BaseLLMPlanner:
    def decompose(self, prompt: str) -> List[Dict]:
        raise NotImplementedError()


class MockLLMPlanner(BaseLLMPlanner):
    def decompose(self, prompt: str) -> List[Dict]:
        # Very simple deterministic decomposition for predictable behavior
        raw = [s.strip() for s in prompt.replace('?', '.').split('.') if s.strip()]
        out = []
        for i, s in enumerate(raw, start=1):
            out.append({"id": i, "action": s + " (from-llm)", "tool": "sample_tool", "args": {"query": s}})
        return out


def get_llm_planner():
    # If OpenAI key available and openai package present, return a simple OpenAI-based planner
        if os.environ.get("OPENAI_API_KEY") and openai is not None and jsonschema_validate is not None:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        class OpenAIPlanner(BaseLLMPlanner):
            def decompose(self, prompt: str) -> List[Dict]:
                # Ask the model to return a JSON array of steps with id, action, tool, args
                system = "Return a JSON array of steps. Each step should be an object with keys: id (int), action (string), tool (string), args (object). Do not include extra text."
                try:
                    resp = openai.ChatCompletion.create(
                        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=512,
                        temperature=0.0,
                    )
                    text = resp["choices"][0]["message"]["content"]
                    import json, re
                    # Try to extract first JSON array from model output
                    m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", text)
                    candidate = None
                    if m:
                        candidate = m.group(1)
                    else:
                        candidate = text.strip()
                    data = json.loads(candidate)
                    # validate schema
                    jsonschema_validate(instance=data, schema=STEPS_ARRAY_SCHEMA)
                    return data
                except Exception:
                    # on any parse/validation error, fall back to mock deterministic planner
                    return MockLLMPlanner().decompose(prompt)

        return OpenAIPlanner()

    return MockLLMPlanner()
