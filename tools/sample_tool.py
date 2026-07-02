"""A tiny example tool implementing `run(args)`.

This simulates a document fetch / summarization tool.
"""
from typing import Dict


def run(args: Dict):
    query = args.get("query", "")
    # Very small deterministic behaviour for demo/testability
    return {
        "query": query,
        "summary": f"(simulated) Found 2 items about: {query[:120]}",
        "metadata": {"source_count": 2},
    }
