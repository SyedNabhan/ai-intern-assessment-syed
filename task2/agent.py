# agent.py
import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END
import langfuse

from agent_state import AgentState, default_state
from agent_tools import web_search, scrape_page, query_rag, format_report

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

lf = langfuse.Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════
# NODE 1 — INPUT VALIDATOR
# ══════════════════════════════════════════════════════════════

def input_validator(state: AgentState) -> dict:
    print(">> [Node 1] input_validator")
    query = state.get("query", "").strip()

    if not query:
        return {"error_message": "Query is empty.", "is_complete": True}

    alpha_chars = [c for c in query if c.isalpha()]
    if len(alpha_chars) == 0:
        print("   REJECTED: no letters")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    vowels = sum(c in "aeiouAEIOU" for c in alpha_chars)
    vowel_ratio = vowels / len(alpha_chars)
    if vowel_ratio < 0.15:
        print("   REJECTED: gibberish")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    words = query.split()
    if all(len(set(w.lower())) <= 1 for w in words if w.isalpha()):
        print("   REJECTED: repeated letters")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    if len(words) == 1:
        print("   REJECTED: too vague")
        return {
            "needs_clarification": True,
            "clarification_question": f"'{query}' is too vague. What specifically about '{query}' do you want to research?",
            "is_complete": True,
        }

    if len(query) < 5:
        return {"error_message": "Query is too short.", "is_complete": True}

    print(f"   Query accepted: {query[:60]}")
    return {}


# NODE 2 — PLANNER
# ══════════════════════════════════════════════════════════════

def planner(state: AgentState) -> dict:
    print(">> [Node 2] planner")

    if state.get("is_complete"):
        return {}

    iteration = state.get("iteration_count", 0)
    search_results = state.get("search_results", [])

    # Hard stop conditions — don't even call LLM
    if iteration >= 3 or len(search_results) >= 3:
        print("   Planner: enough data, marking complete")
        return {
            "is_complete": True,
            "iteration_count": iteration + 1,
        }

    # Always do web_search — simple and reliable
    query = state.get("query", "")
    print(f"   Planner: doing web_search for '{query[:50]}'")
    return {
        "current_tool": "web_search",
        "tool_args": {"query": query},
        "iteration_count": iteration + 1,
    }

# ══════════════════════════════════════════════════════════════
# NODE 3 — TOOL CALLER
# ══════════════════════════════════════════════════════════════

def tool_caller(state: AgentState) -> dict:
    print(">> [Node 3] tool_caller")

    if state.get("is_complete"):
        return {}

    tool = state.get("current_tool")
    args = state.get("tool_args", {})

    print(f"   Calling: {tool} with args: {args}")

    try:
        if tool == "web_search":
            results = web_search(args.get("query", state.get("query")))
            print(f"   Got {len(results)} search results")
            existing = state.get("search_results", [])
            return {"search_results": existing + results}

        elif tool == "scrape_page":
            result = scrape_page(args.get("url", ""))
            if result["error"]:
                existing_errors = state.get("tool_errors", [])
                return {"tool_errors": existing_errors + [f"scrape failed: {result['error']}"]}
            existing = state.get("scraped_pages", [])
            return {"scraped_pages": existing + [result]}

        elif tool == "query_rag":
            result = query_rag(args.get("question", state.get("query")))
            if result["error"]:
                existing_errors = state.get("tool_errors", [])
                return {"tool_errors": existing_errors + [f"RAG error: {result['error']}"]}
            existing = state.get("rag_results", [])
            return {"rag_results": existing + [result]}

        else:
            existing_errors = state.get("tool_errors", [])
            return {"tool_errors": existing_errors + [f"Unknown tool: {tool}"]}

    except Exception as e:
        print(f"   Tool crashed: {e}")
        existing_errors = state.get("tool_errors", [])
        return {"tool_errors": existing_errors + [f"{tool} crashed: {str(e)}"]}


# ══════════════════════════════════════════════════════════════
# NODE 4 — SYNTHESIZER
# ══════════════════════════════════════════════════════════════

def synthesizer(state: AgentState) -> dict:
    print(">> [Node 4] synthesizer")

    if state.get("error_message") or state.get("needs_clarification"):
        print("   Skipping synthesis — error or clarification needed")
        return {}

    rag = state.get("rag_results", [])
    raw_report = format_report(
        query=state.get("query", ""),
        search_results=state.get("search_results", []),
        scraped_pages=state.get("scraped_pages", []),
        rag_results=rag[0] if rag else {"results": []},
        tool_errors=state.get("tool_errors", []),
    )

    prompt = f"""You are a research report writer.

Original question: {state.get("query")}

Raw research data:
{raw_report[:3000]}

Write a clean structured research report with:
1. A short summary answering the question
2. Key findings as bullet points  
3. Sources cited at the bottom

Use markdown formatting. Be concise."""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3,
    )

    report = response.choices[0].message.content.strip()
    print(f"   Report written: {len(report)} chars")
    return {"final_report": report}


# ══════════════════════════════════════════════════════════════
# CONDITIONAL EDGE
# ══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    iteration = state.get("iteration_count", 0)
    search_results = state.get("search_results", [])
    
    print(f">> [Edge] iteration={iteration} search_results={len(search_results)}")

    if iteration >= 3:
        print(">> [Edge] iteration limit → synthesizer")
        return "synthesizer"
    if len(search_results) >= 3:
        print(">> [Edge] enough results → synthesizer")
        return "synthesizer"
    if state.get("total_cost_usd", 0) >= state.get("cost_limit_usd", 1.0):
        print(">> [Edge] cost limit → synthesizer")
        return "synthesizer"
    if state.get("is_complete"):
        print(">> [Edge] complete → synthesizer")
        return "synthesizer"

    print(">> [Edge] continuing → tool_caller")
    return "tool_caller"


# ══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("input_validator", input_validator)
    builder.add_node("planner",         planner)
    builder.add_node("tool_caller",     tool_caller)
    builder.add_node("synthesizer",     synthesizer)

    builder.add_edge(START, "input_validator")
    builder.add_edge("input_validator", "planner")
    builder.add_edge("planner", "tool_caller")
    builder.add_conditional_edges(
        "tool_caller",
        should_continue,
        {
            "tool_caller": "tool_caller",
            "synthesizer": "synthesizer",
        }
    )
    builder.add_edge("synthesizer", END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════
# SAVE RUN TO FILE
# ══════════════════════════════════════════════════════════════

def save_run(query: str, filename: str):
    graph = build_graph()
    initial = default_state(query=query)

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")

    with lf.start_as_current_observation(
        as_type="agent",
        name="research-agent-run",
        input={"query": query},
        metadata={"filename": filename},
    ):
        final_state = graph.invoke(initial, config={"recursion_limit": 25})

        with lf.start_as_current_observation(
            as_type="span",
            name="final-output",
            input={
                "final_report": final_state.get("final_report"),
                "error": final_state.get("error_message"),
                "iterations": final_state.get("iteration_count"),
            },
        ):
            pass

    lf.flush()
    time.sleep(1)

    # Save to runs/ folder
    os.makedirs("runs", exist_ok=True)
    run_data = {
        "query": query,
        "final_report": final_state.get("final_report"),
        "error_message": final_state.get("error_message"),
        "needs_clarification": final_state.get("needs_clarification"),
        "clarification_question": final_state.get("clarification_question"),
        "iterations": final_state.get("iteration_count"),
        "search_results_count": len(final_state.get("search_results", [])),
        "tool_errors": final_state.get("tool_errors", []),
    }

    with open(f"runs/{filename}.json", "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"\n{'='*60}")
    if final_state.get("error_message"):
        print(f"ERROR: {final_state.get('error_message')}")
    elif final_state.get("needs_clarification"):
        print(f"CLARIFICATION NEEDED: {final_state.get('clarification_question')}")
    else:
        print("FINAL REPORT:")
        print(final_state.get("final_report"))
    print(f"\nIterations: {final_state.get('iteration_count')}")
    print(f"Saved to: runs/{filename}.json")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run 1 — happy path
    save_run(
        "What are the latest developments in large language models?",
        "run_1_happy_path"
    )