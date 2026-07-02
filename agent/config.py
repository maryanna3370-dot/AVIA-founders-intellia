"""Agent configuration and feature flags."""
import os

# Tools allowed by default (None means allow all)
ALLOWED_TOOLS = os.environ.get("ALLOWED_TOOLS")
if ALLOWED_TOOLS:
    ALLOWED_TOOLS = [t.strip() for t in ALLOWED_TOOLS.split(",") if t.strip()]

# Use LLM planner when OPENAI_API_KEY or LLM_PROVIDER is set
USE_LLM = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_PROVIDER"))

# Redis URL for optional memory persistence
REDIS_URL = os.environ.get("REDIS_URL")

# Sandboxing: run tools in separate process
SANDBOX_TOOLS = os.environ.get("SANDBOX_TOOLS", "1") != "0"
# Sandbox mode: 'subprocess' or 'docker'
SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "subprocess")
# Docker image to use for sandboxed execution (must have Python and repo mounted)
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "python:3.12-slim")

# Default maximum wait for approval (seconds)
DEFAULT_APPROVAL_WAIT = int(os.environ.get("DEFAULT_APPROVAL_WAIT", "30"))
