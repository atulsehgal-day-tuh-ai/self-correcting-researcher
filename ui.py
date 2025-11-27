import streamlit as st
import os
from dotenv import load_dotenv

# --- CRITICAL FIX: LOAD KEYS FIRST ---
# This must happen BEFORE importing src.graph
load_dotenv()
# -------------------------------------

from langchain_core.tracers.context import collect_runs
from langsmith import Client
from src.graph import build_graph


# 2. Setup LangSmith Client (to generate URLs)
client = Client()

st.set_page_config(page_title="Self-Correcting Researcher", page_icon="🤖")

st.title("🤖 Self-Correcting Researcher")
st.markdown("Ask a question. If the agent finds bad results, it will **self-correct** and rewrite your query.")

# 3. Input Form
with st.form("agent_form"):
    text = st.text_area("Enter your question:", "How does agent memory work?")
    submitted = st.form_submit_button("Submit")

if submitted and text:
    # Initialize the Graph
    app = build_graph()
    inputs = {"question": text, "retry_count": 0}
    
    # Create a placeholder for the "Thinking..." stream
    status_container = st.status("Thinking...", expanded=True)
    
    final_generation = ""
    run_id = None

    # --- THE MAGIC PART: collect_runs() ---
    # This context manager captures the trace info automatically
    with collect_runs() as cb:
        
        # Run the graph stream
        for output in app.stream(inputs):
            for key, value in output.items():
                
                # Update the UI Waterfall
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
        
        # Capture the Run ID from the first tracked run (the root)
        if cb.traced_runs:
            run_id = cb.traced_runs[0].id

    status_container.update(label="Finished!", state="complete", expanded=False)

    # 4. Display Result
    st.subheader("Answer")
    st.write(final_generation)

    # 5. Display LangSmith Link
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