"""
LangGraph workflow definition.

This module wires together the core nodes in `src/nodes.py` into a stateful graph
with a self-correction loop:

    retrieve -> grade_documents -> (generate OR rewrite_query) -> retrieve -> ...

The loop continues until:
- at least one relevant document is found (generate), OR
- a retry limit is exceeded (generate anyway with whatever context we have)

The state that flows through the graph is defined by `src.state.GraphState`.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.nodes import generate, grade_documents, retrieve, rewrite_query
from src.state import GraphState

# Safety valve: prevents an infinite loop if retrieval keeps failing.
MAX_RETRIES = 3


def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        A compiled LangGraph application that can be executed via:
        - `app.invoke(inputs)` for a one-shot call
        - `app.stream(inputs)` to observe node-by-node state updates
    """

    workflow = StateGraph(GraphState)

    # --- Nodes ---
    # Node names are what you see in LangSmith traces (and in Streamlit waterfall).
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)

    # --- Edges / Control flow ---
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional edge: after grading, decide whether we can answer or must retry.
    def decide_to_generate(state: GraphState) -> str:
        """
        Decide next step after grading documents.

        Returns one of the keys in the conditional edge map below:
        - "generate": answer now
        - "rewrite_query": rewrite and retry retrieval
        """

        # If any docs survived the grader filter, answer using them.
        if state["documents"]:
            return "generate"

        # Otherwise: if we have retried enough times, stop looping and answer anyway.
        if state.get("retry_count", 0) > MAX_RETRIES:
            return "generate"

        # Retry path: rewrite and search again.
        return "rewrite_query"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # Loop back to retrieval after rewriting.
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()