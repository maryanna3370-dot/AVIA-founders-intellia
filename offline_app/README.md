# Offline AI App

This is a minimal offline AI app that runs a local Transformers model and exposes a small web UI + JSON endpoint.

Quick start (first run will download the model if needed):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r offline_app/requirements.txt
python offline_app/app.py
```

Then open http://localhost:7860/ and enter a prompt.

Notes:
- By default the app uses `distilgpt2`. To point to a local model folder, set `MODEL_PATH` env var.
- Models are large; download requires internet on first run. To run fully offline, pre-download a model into the HF cache or set `MODEL_PATH` to a local directory with model files.
