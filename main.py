"""
CLI entrypoint for the Self-Correcting Researcher.

This script runs the LangGraph workflow in the terminal (no Streamlit UI).
It is useful for quick smoke tests and for watching the loop behavior in logs.

How it works (high level):
- Loads environment variables from `.env` (OpenAI key, LangSmith tracing config, etc.).
- Builds the LangGraph workflow via `src.graph.build_graph()`.
- Streams node-by-node outputs using `app.stream(inputs)` until completion.

Tip (Windows):
- Prefer running with the project venv interpreter:
  `.\\.venv\\Scripts\\python.exe main.py`
  This avoids accidentally using a system Python without the pinned dependencies.
"""

from dotenv import load_dotenv

# Load keys and tracing configuration (must happen before importing graph/nodes).
load_dotenv()

from src.graph import build_graph


def main() -> None:
    """Run one example query through the workflow and print the final answer."""
    app = build_graph()

    # GraphState-compatible input. `retry_count` is used to limit loop iterations.
    inputs = {"question": "How does agent memory work?", "retry_count": 0}

    print("\n\n--- STARTING AGENT ---")
    final_state = {}

    # `app.stream()` yields partial state updates per node execution.
    for output in app.stream(inputs):
        for node_name, state_update in output.items():
            final_state = state_update
            print(f"Finished Node: {node_name}")

    print("\n\n--- FINAL RESULT ---")
    print(final_state.get("generation"))


if __name__ == "__main__":
    main()