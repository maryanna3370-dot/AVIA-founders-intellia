"""FastAPI application entrypoint for the AVIA founders Intellia agent web UI.

Run this file to start the local web interface for enqueuing agent tasks
and inspecting pending task status.
"""

from agent.webapp import app

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run app.py. Install it with `pip install uvicorn`."
        ) from exc

    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info")
