import os
import math
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── 1. Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluates a mathematical expression and returns the numeric result. "
            "Use this for arithmetic, percentages, or any calculation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '2 + 3 * 4' or 'math.sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web for information and returns mock results. "
            "Use this to look up facts, current values, or any external information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
]

# ── 2. Tool implementations ────────────────────────────────────────────────────

def run_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def run_web_search(query: str) -> str:
    """Mock search — returns realistic-looking stub data keyed to common queries."""
    mock_db = {
        "bitcoin": "Mock result: Bitcoin (BTC) current price is $67,432 USD as of today.",
        "gold":    "Mock result: Gold spot price is $2,345 per troy ounce.",
        "apple":   "Mock result: Apple Inc. (AAPL) stock price is $189.50.",
        "population": "Mock result: World population is approximately 8.1 billion people.",
        "gdp":     "Mock result: US GDP is approximately $27.4 trillion USD.",
    }
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key in query_lower:
            return value
    return f"Mock result: No specific data found for '{query}'. Estimated value: 42 (placeholder)."


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "calculator":
        return run_calculator(tool_input["expression"])
    if name == "web_search":
        return run_web_search(tool_input["query"])
    return f"Unknown tool: {name}"


# ── 3–6. Agentic loop ──────────────────────────────────────────────────────────

MAX_ITERATIONS = 20  # safety cap — should never be reached in normal operation


def run_agent(user_prompt: str) -> str:
    """
    Runs the agentic loop until Claude signals end_turn or the safety cap fires.
    Returns Claude's final text response.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": user_prompt}]
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n[iteration {iteration}] Sending request to Claude…")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        stop_reason = response.stop_reason
        print(f"[iteration {iteration}] stop_reason = {stop_reason!r}")

        # ── 4. end_turn → extract final text and exit ──────────────────────────
        if stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # ── 3. tool_use → execute tool and append result ───────────────────────
        if stop_reason == "tool_use":
            # Append Claude's full response (may contain text + tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"[iteration {iteration}] Tool called: {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    print(f"[iteration {iteration}] Tool result : {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            continue  # go back to top of loop

        # Unexpected stop_reason — break to avoid infinite loop
        print(f"[warning] Unexpected stop_reason: {stop_reason!r}. Exiting loop.")
        break

    # ── 6. Safety cap triggered ────────────────────────────────────────────────
    print(f"[WARNING] Safety iteration cap of {MAX_ITERATIONS} reached — loop terminated early.")
    # Return whatever text Claude produced in the last response, if any
    for block in response.content:
        if block.type == "text":
            return block.text
    return "Agent stopped: iteration cap reached."


# ── 5. Multi-tool test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    prompt = (
        "Search for the current price of Bitcoin, then calculate how many "
        "whole Bitcoins I could buy with $500,000 USD."
    )
    print(f"User: {prompt}")
    answer = run_agent(prompt)
    print("\nFinal answer:")
    print(answer.encode("cp1252", errors="replace").decode("cp1252"))
