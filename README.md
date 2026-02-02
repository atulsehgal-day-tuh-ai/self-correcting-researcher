## Self-Correcting Researcher (RAG + LangGraph + LangSmith + Streamlit)

This repo implements a **self-correcting RAG agent**:
- It retrieves candidate context from a persistent vector store (Chroma).
- It uses an LLM to **grade retrieval quality**.
- If retrieval is bad, it **rewrites the query** and tries again (bounded retries).
- Once enough relevant context exists (or we hit the retry limit), it **generates** a final answer.

The implementation is intentionally small so it’s easy to study and extend.

### Quick links
- Architecture & flow diagrams: `diagram.md`
- Streamlit UI: `ui.py`
- CLI runner: `main.py`
- Graph wiring: `src/graph.py`
- Node logic: `src/nodes.py`
- State model: `src/state.py`

---

## Project structure

```text
self-correcting-researcher/
├── chroma_db/              # Persistent Vector DB (generated at runtime; not committed)
├── src/
│   ├── graph.py            # The "Brain": Defines the workflow and decision logic
│   ├── nodes.py            # The "Muscles": Actual functions (Search, Grade, Write)
│   └── state.py            # The "Memory": Data passed between steps
├── ui.py                   # Streamlit Frontend
├── main.py                 # CLI Execution script
├── .env                    # API Keys (Not committed)
├── pyproject.toml          # Dependencies
└── README.md               # Documentation
```

---

## Architecture & code flow

The workflow is defined in `src/graph.py` and uses four nodes implemented in `src/nodes.py`:

- **retrieve**: get candidate chunks from Chroma
- **grade_documents**: LLM relevance filter (per-chunk yes/no)
- **rewrite_query**: query optimizer used only when retrieval is bad
- **generate**: final answer generation (RAG)

See `diagram.md` for diagrams and detailed notes.

---

## How to run

### Prerequisites
- Python 3.12.x (repo is pinned to 3.12 in `pyproject.toml`)
- OpenAI API key
- LangSmith API key (optional, but recommended)

### 1. Installation
Clone the repository and install dependencies using `uv` (recommended) or `pip`.

```bash
# Using uv (Fast)
uv sync

# OR using standard pip
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=sk-proj-your-key...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2-your-key...
LANGCHAIN_PROJECT="Self-Correcting-Researcher"
```

### 3. Execution

#### Option A: Web UI (Streamlit)
This provides a visual interface with a "Thinking Waterfall" and direct links to LangSmith traces.

```bash
streamlit run ui.py
```

If `streamlit run ui.py` fails on Windows with **"Failed to canonicalize script path"**, use:

```bash
python -m streamlit run .\ui.py
```

#### Option B: CLI (Terminal)
Runs the agent in the command line for quick testing.

```bash
python main.py
```

---

## LangSmith (tracing + how to interpret runs)

If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, LangSmith will capture a trace per run.

### What you’ll see in the trace
- A top-level **LangGraph** run for the whole execution
- Child runs for each node (`retrieve`, `grade_documents`, `rewrite_query`, `generate`)
- Nested runs inside nodes (e.g., `VectorStoreRetriever`, `ChatOpenAI`)

### Debug fields emitted by this repo (metadata-only)
To make the agent behavior easier to understand, node outputs include additional **metadata-only** fields:
- `retrieved_docs_meta` (from `retrieve`): per-doc metadata (often includes `source`)
- `doc_grades` (from `grade_documents`): per-doc pass/fail + grade + metadata
- `used_docs_meta` (from `generate`): metadata for docs actually used in the final answer

These appear in the **Outputs** panel when you click each node run in LangSmith.

---

## Troubleshooting

**Blank Screen in UI?**
If the Streamlit app stays blank, the vector database might be downloading in the background. Check your terminal for progress logs.

**Database Corrupted/Empty Answers?**
If you stopped the script mid-creation, `chroma_db` might be corrupted.
1.  Stop the app (`Ctrl+C`).
2.  Delete the `chroma_db/` folder.
3.  Restart the app to force a rebuild.

**Import Errors?**
Ensure your virtual environment is active and selected in VS Code (`Ctrl+Shift+P` -> `Python: Select Interpreter`).