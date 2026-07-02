"""Filesystem search tool for the workspace.

Args: {"query":"text", "max_results": 10}
Returns: list of {path, snippet}
"""
from typing import Dict, Any
import os


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query", "")
    if not query:
        return {"status": "error", "error": "missing_query"}

    max_results = int(args.get("max_results", 10))
    root = args.get("root", ".")
    matches = []
    q = query.lower()
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if q in text.lower():
                    idx = text.lower().index(q)
                    start = max(0, idx - 40)
                    snippet = text[start: start + 160].replace('\n', ' ')
                    matches.append({"path": path, "snippet": snippet})
                    if len(matches) >= max_results:
                        return {"status": "ok", "matches": matches}
            except Exception:
                continue

    return {"status": "ok", "matches": matches}
