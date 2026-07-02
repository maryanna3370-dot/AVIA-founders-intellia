# AI Founders Toolkit

**100% Offline AI knowledge and reasoning platform** – Collect feedback, store structured knowledge, perform semantic retrieval, and get AI-generated insights with source citations.

## Features

- ✅ **Stage 1: Offline Desktop Utility** – Feedback collection from 4 stakeholders on business problems
- ✅ **Stage 2: Persistent Knowledge System** – SQLite database with local vector embeddings  
- ✅ **Stage 3: Semantic Retrieval Engine** – Query embedding + cosine similarity search over reports
- ✅ **Stage 4: Local RAG Assistant** – Question answering with source citations (no API calls)

### Input Methods

- Direct text entry
- PDF files (PyPDF2)
- JSON structured data (feedback, notes)
- WAV audio files (SpeechRecognition + pocketsphinx, offline)
- Microphone recording (offline pocketsphinx)

### Knowledge Management

- Collects feedback from 4 stakeholders per session
- Saves JSON report exports to `reports/`
- Stores all records in SQLite at `reports/reports.db`
- Generates deterministic local text vectors for semantic search
- Supports multi-source import (JSON, PDF) for bulk ingestion

### AI Capabilities

- **Search & Summarize** – Retrieve relevant reports and generate insight responses
- **RAG Answer** – Generate citations with source filenames, IDs, and similarity scores
- **Configurable retrieval** – Adjust Top-K parameter (1-10 sources)
- **Local summarization** – BART or fallback sentence extraction (no external LLM)

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Required Python packages

The project depends on the following Python packages:

- PyPDF2>=3.0.0
- SpeechRecognition>=3.8.1
- pocketsphinx>=0.1.15
- numpy>=1.24.0
- scikit-learn>=1.4.0
- sentence-transformers>=2.3.0
- transformers>=4.40.0
- torch>=2.0.0
- faiss-cpu>=1.7.4
- requests>=2.30.0
- fastapi>=0.95.0
- uvicorn>=0.22.0
- redis>=4.6.0
- psycopg2-binary>=2.9.8
- openai>=1.0.0
- jsonschema>=4.19.0

## Speech recognition setup

This app supports offline speech recognition for WAV files and microphone input using `SpeechRecognition` + `pocketsphinx`.

Install the required speech packages:

```bash
python3 -m pip install SpeechRecognition pocketsphinx
```

Then you can:

- load WAV files through the GUI and convert speech to text
- record live microphone audio and insert recognized text into the problem input
- use the continuous voice conversation flow for spoken queries

If the app cannot find `SpeechRecognition`, it shows an error like `SpeechRecognition not installed`.

## Run the app

Start the application from the repository root:

```bash
python3 ai_founders_intellia.py
```

## Use a custom workspace directory

To keep `reports/` and `reports.db` in a different folder:

```bash
python3 ai_founders_intellia.py --workspace-root /path/to/workspace
```

Or set the environment variable:

```bash
export AI_WORKSPACE_ROOT=/path/to/workspace
python3 ai_founders_intellia.py
```

## Optional Ollama integration

If you want the app to use a local Ollama model for AI responses:

```bash
ollama pull llama2
export OLLAMA_MODEL_NAME=llama2
```

The app will automatically use Ollama when the CLI and model are available, and fall back to the local transformer pipeline or summarization otherwise.

## Useful CLI commands

Inspect a specific report chunk and its neighboring chunks:

```bash
python3 ai_founders_intellia.py --chunk-with-neighbors <REPORT_ID> <CHUNK_NUMBER>
```

Example:

```bash
python3 ai_founders_intellia.py --chunk-with-neighbors 1 7
```

Run the autonomous agent with a query:

```bash
python3 ai_founders_intellia.py --workspace-root /workspaces/AVIA-founders-intellia --agent --agent-query "Summarize recent reports about fundraising"
```

Enqueue reports for background indexing only:

```bash
python3 ai_founders_intellia.py --workspace-root /workspaces/AVIA-founders-intellia --agent
```

## Workflow

1. **Enter or import** a business problem (text, PDF, JSON, or voice)
2. **Collect feedback** from 4 stakeholders
3. **Save report** – stored in JSON and SQLite with embeddings
4. **Ask questions** and get AI-generated insights with cited sources
5. **Adjust Top-K** parameter to control retrieval depth

## Output

- JSON exports in `reports/`
- SQLite database at `reports/reports.db`
- Vector embeddings stored in SQLite for similarity search
- RAG answers include source citations (filename, report ID, similarity score)

## Notes

- Fully offline – no internet or cloud dependencies
- All data stays local on your machine
- Embeddings use deterministic hashing for consistent, reproducible similarity
- RAG summaries use local sentence extraction or BART if available

## Agent (CLI)

Use the built-in autonomous CLI agent to reindex reports and run a streaming RAG query from the command line.

- Reindex reports and run a query (reindexes JSON files in `reports/`, starts the background indexer, then streams the RAG answer):

```bash
python3 ai_founders_intellia.py --workspace-root /workspaces/AVIA-founders-intellia --agent --agent-query "Summarize recent reports about fundraising"
```

- Enqueue all JSON reports for background indexing only:

```bash
python3 ai_founders_intellia.py --workspace-root /workspaces/AVIA-founders-intellia --agent
```

Notes:
- Set `AI_WORKSPACE_ROOT` instead of `--workspace-root` to change the data directory.
- When Ollama is available, RAG answers stream incrementally; otherwise the agent will use the local pipeline/fallback.
- Agent output is printed to STDOUT with lightweight progress markers prefixed by `[AGENT]` and `[AGENT INDEX]`.
