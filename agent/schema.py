"""JSON schema definitions for planner/executor step objects."""
STEP_SCHEMA = {
    "type": "object",
    "required": ["id", "action", "tool", "args"],
    "properties": {
        "id": {"type": "integer"},
        "action": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "delay_s": {"type": "number"},
        "max_retries": {"type": "integer"},
    },
    "additionalProperties": True,
}

STEPS_ARRAY_SCHEMA = {"type": "array", "items": STEP_SCHEMA}
