"""Generic HTTP API tool adapter.

Expects args: {"method": "GET", "url": "...", "params":{}, "headers":{}, "json":{}, "timeout":5}
Returns a standardized JSON-compatible dict.
"""
from typing import Dict, Any
import requests


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    method = args.get("method", "GET").upper()
    url = args.get("url")
    if not url:
        return {"status": "error", "error": "missing_url"}

    params = args.get("params")
    headers = args.get("headers")
    json_body = args.get("json")
    timeout = args.get("timeout", 10)

    try:
        resp = requests.request(method, url, params=params, headers=headers, json=json_body, timeout=timeout)
        text = resp.text
        return {
            "status": "ok",
            "status_code": resp.status_code,
            "text": text[:10000],
            "headers": dict(resp.headers),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
