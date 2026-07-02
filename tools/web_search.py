"""Simple web search tool with safe default (simulation).

By default the tool simulates results to avoid network calls. Set `live=True` in args to perform a live DuckDuckGo HTML search.
"""
from typing import Dict, Any
import requests
import html
import re


def _parse_duckduckgo(html_text: str):
    # Extremely small, best-effort parser for DuckDuckGo HTML.
    hits = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', html_text, re.S):
        title = re.sub(r'<.*?>', '', m.group(1)).strip()
        hits.append({"title": html.unescape(title), "snippet": ""})
        if len(hits) >= 5:
            break
    return hits


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query", "")
    if not query:
        return {"status": "error", "error": "missing_query"}

    live = bool(args.get("live", False))
    if not live:
        # Simulated deterministic response for offline use and tests
        return {
            "status": "ok",
            "results": [
                {"title": f"Simulated result A for {query}", "snippet": "Example snippet A"},
                {"title": f"Simulated result B for {query}", "snippet": "Example snippet B"},
            ],
        }

    try:
        url = "https://html.duckduckgo.com/html/"
        resp = requests.get(url, params={"q": query}, timeout=8)
        parsed = _parse_duckduckgo(resp.text)
        return {"status": "ok", "results": parsed}
    except Exception as e:
        return {"status": "error", "error": str(e)}
