# Minimal Planner–Executor Agent

This folder contains a minimal, runnable planner->executor prototype for experimenting with autonomous workflows.

Run (interactive approvals):

```bash
python -m agent.cli --task "Summarize market sizing. Find competitors."
```

Run (auto-approve):

```bash
python -m agent.cli --task "Summarize market sizing. Find competitors." --approve
```

The `Planner` decomposes the prompt into simple steps. The `Executor` dynamically imports tools from the `tools` package and calls `run(args)` on them.

Included tools (examples):

- `tools/sample_tool.py` — deterministic demo tool used by default.
- `tools/fs_search.py` — search repository files for a query string.
- `tools/web_search.py` — simulated web search (set `live=True` to perform a DuckDuckGo query).
- `tools/http_api.py` — generic HTTP adapter to call external APIs.

Tool contract: each tool module must expose `run(args: dict) -> dict`.
