from langgraph.graph import END, StateGraph
from src.state import GraphState
from src.nodes import retrieve, generate, grade_documents, rewrite_query

def build_graph():
    workflow = StateGraph(GraphState)

    # 1. Define Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)

    # 2. Define Edges (The Logic)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional Logic: Decide where to go after grading
    def decide_to_generate(state):
        # If we have relevant documents, generate answer
        if state["documents"]:
            return "generate"
        # If we have tried too many times (loop limit), just give up and generate with what we have
        elif state.get("retry_count", 0) > 3:
            return "generate"
        # Otherwise, rewrite the query and search again
        else:
            return "rewrite_query"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve") # Loop back to search
    workflow.add_edge("generate", END)

    return workflow.compile()