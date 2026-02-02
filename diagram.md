## Architecture & Flow

This doc is the single source of truth for how the repo fits together.

### High-level control flow (LangGraph)

```mermaid
flowchart TD
    StartNode["Start (main.py/ui.py): app.stream(inputs)"] --> RetrieveNode["retrieve (src/nodes.py)"]
    RetrieveNode --> GradeNode["grade_documents (src/nodes.py)"]
    GradeNode --> DecideNode{"decide_to_generate (src/graph.py)"}

    DecideNode -->|"documents non-empty"| GenerateNode["generate (src/nodes.py)"]
    DecideNode -->|"documents empty AND retry_count <= MAX_RETRIES"| RewriteNode["rewrite_query (src/nodes.py)"]
    DecideNode -->|"documents empty AND retry_count > MAX_RETRIES"| GenerateNode

    RewriteNode --> RetrieveNode
    GenerateNode --> EndNode["End (final state)"]
```

### Module responsibilities

```mermaid
flowchart LR
    Main["main.py (CLI)"] --> Graph["src/graph.py build_graph()"]
    UI["ui.py (Streamlit)"] --> Graph

    Graph --> Nodes["src/nodes.py (node fns)"]
    Graph --> State["src/state.py (GraphState)"]

    Nodes --> Retriever["Chroma Retriever (persisted ./chroma_db)"]
    Nodes --> LLM["ChatOpenAI (gpt-4o-mini)"]

    UI --> LangSmith["LangSmith (collect_runs + tracing)"]
    Nodes --> LangSmith
```

### State (GraphState) fields that drive behavior

- `question`: original user question or rewritten query
- `documents`: retrieved docs (then filtered by `grade_documents`)
- `retry_count`: increments when *no* relevant docs are found, used to stop infinite loops
- `generation`: final answer (set by `generate`)

### Trace/debug fields (LangSmith-friendly, metadata-only)

These fields are returned by nodes so you can inspect behavior in LangSmith without logging raw text:

- `retrieved_docs_meta` (from `retrieve`): per-doc metadata (e.g., `source` URL) + index
- `doc_grades` (from `grade_documents`): per-doc pass/fail + grade + metadata
- `used_docs_meta` (from `generate`): metadata for the final docs used to answer