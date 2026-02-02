"""
Streamlit UI for the Self-Correcting Researcher.

This provides:
- a simple question input box
- a "thinking waterfall" status area that updates as graph nodes run
- a link to the LangSmith trace (via `collect_runs()` + LangSmith client)

Notes:
- `.env` must be loaded *before* importing `src.graph` / `src.nodes` so that
  OpenAI/LangSmith environment variables are available during initialization.
- On some Windows setups, `streamlit run ui.py` can fail with
  "Failed to canonicalize script path". If you hit that, run:
  `python -m streamlit run .\\ui.py`
"""

from __future__ import annotations

import streamlit as st

from dotenv import load_dotenv

# --- Critical: load keys first ---
# This must happen BEFORE importing `src.graph`, because `src.nodes` initializes
# the OpenAI embedding/model objects at import time.
load_dotenv()

from langchain_core.tracers.context import collect_runs
from langsmith import Client

from src.graph import build_graph


# Setup LangSmith Client (to generate trace URLs).
client = Client()

def _select_root_run_id(traced_runs) -> str | None:
    """
    Pick the *root* LangSmith run ID from a `collect_runs()` context.

    Why this exists:
    - A single graph execution produces MANY runs (root + node runs + nested LLM/retriever runs).
    - LangSmith stores these runs as a *tree* via parent/child relationships.
    - To link to “the whole trace”, we want the ROOT run (the run with no parent).

    What `collect_runs()` gives us:
    - `cb.traced_runs` is a list of run objects created during the context.
    - Depending on what ran inside the context, this list may contain:
      - multiple child runs
      - potentially multiple independent root runs (if you ran multiple graphs/chains)

    Strategy:
    1) Prefer runs that have no parent (`parent_run_id is None`).
    2) If multiple roots exist, prefer the one named "LangGraph" (common for LangGraph root).
    3) Otherwise, fall back to the earliest root (or first captured run if no parent info exists).
    """
    if not traced_runs:
        return None

    # Root runs usually have no parent_run_id (None). Some objects may not expose this field.
    roots = [r for r in traced_runs if getattr(r, "parent_run_id", None) in (None, "")]

    if not roots:
        # Fallback: if parent metadata isn't present, use the first captured run.
        return getattr(traced_runs[0], "id", None)

    # Prefer the LangGraph root run if present.
    for r in roots:
        if getattr(r, "name", "") == "LangGraph":
            return getattr(r, "id", None)

    # Otherwise: prefer earliest root (best-effort), else just take the first.
    def _sort_key(r):
        return getattr(r, "start_time", None) or ""

    roots_sorted = sorted(roots, key=_sort_key)
    return getattr(roots_sorted[0], "id", None)

st.set_page_config(page_title="Self-Correcting Researcher", page_icon="🤖")

st.title("🤖 Self-Correcting Researcher")
st.markdown("Ask a question. If the agent finds bad results, it will **self-correct** and rewrite your query.")

# Input form (keeps the UI clean and avoids rerunning on every keystroke).
with st.form("agent_form"):
    text = st.text_area("Enter your question:", "How does agent memory work?")
    submitted = st.form_submit_button("Submit")

if submitted and text:
    # Initialize the compiled LangGraph app.
    app = build_graph()
    inputs = {"question": text, "retry_count": 0}
    
    # Placeholder for the "Thinking..." stream (our UI waterfall).
    status_container = st.status("Thinking...", expanded=True)
    
    final_generation = ""
    run_id = None

    # `collect_runs()` does NOT decide what is traced.
    # Tracing happens automatically when LANGCHAIN_TRACING_V2 is enabled.
    #
    # `collect_runs()` is just a convenience wrapper that collects *run objects*
    # created inside this context so we can extract a root run id and link to it.
    with collect_runs() as cb:
        
        # Run the graph stream and update the UI as nodes finish.
        for output in app.stream(inputs):
            for key, value in output.items():
                
                # Update the UI waterfall based on node name.
                if key == "retrieve":
                    status_container.write("🔍 Retrieving documents...")
                elif key == "grade_documents":
                    if value.get("retry_count", 0) > inputs["retry_count"]:
                        status_container.write("❌ Search results irrelevant. Retrying...")
                    else:
                        status_container.write("✅ Documents validated.")
                elif key == "rewrite_query":
                    status_container.write(f"✍️ Rewriting query: '{value['question']}'")
                elif key == "generate":
                    status_container.write("💡 Generating final answer...")
                    final_generation = value["generation"]
        
        # IMPORTANT:
        # `cb.traced_runs` may contain many runs. We pick the ROOT run because
        # it expands to the full trace tree in the LangSmith UI.
        run_id = _select_root_run_id(cb.traced_runs)

    status_container.update(label="Finished!", state="complete", expanded=False)

    # Display result.
    st.subheader("Answer")
    st.write(final_generation)

    # Display LangSmith link (debug trace).
    if run_id:
        # Get the URL from LangSmith Client
        url = client.read_run(run_id).url
        st.markdown(f"""
        <a href="{url}" target="_blank">
            <button style="
                background-color: #f0f2f6; 
                border: 1px solid #d0d7de; 
                border-radius: 6px; 
                padding: 5px 10px; 
                cursor: pointer; 
                font-weight: 600;">
                🔍 View Debug Trace in LangSmith
            </button>
        </a>
        """, unsafe_allow_html=True)