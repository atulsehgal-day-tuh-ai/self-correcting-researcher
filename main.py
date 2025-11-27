import os
from dotenv import load_dotenv
# Load keys
load_dotenv()

from src.graph import build_graph

# Build the app
app = build_graph()

# Run with a query
# Try a hard query to force it to "think" and rewrite, e.g., "What is the memory module?"
# (The blog post talks about 'Memory' in agents, but 'module' might force a specific check)

inputs = {"question": "How does agent memory work?", "retry_count": 0}

print("\n\n--- STARTING AGENT ---")
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished Node: {key}")

print("\n\n--- FINAL RESULT ---")
# The final state is the last output yielded
print(value.get("generation"))