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

    # `collect_runs()` captures the root LangGraph run ID so we can link to the
    # trace in LangSmith. This is independent from the node-level tracing that
    # happens automatically when LangChain tracing is enabled.
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
        
        # Capture the Run ID from the first tracked run (the root).
        if cb.traced_runs:
            run_id = cb.traced_runs[0].id

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