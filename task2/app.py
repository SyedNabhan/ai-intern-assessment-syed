"""
Task 2 — Research Assistant Agent UI
Streamlit app with live streaming agent progress
"""

import sys
import os

# Ensure task-2/ root is on the path so agent.py, agent_tools.py etc. are importable
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task2")
import json
import time
import streamlit as st
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f12;
    color: #e2e8f0;
}

/* Header */
.agent-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #7dd3fc;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.agent-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Stream panel */
.stream-panel {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    min-height: 360px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    overflow-y: auto;
}

/* Node event pill */
.node-pill {
    display: inline-block;
    background: #1e293b;
    color: #7dd3fc;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.tool-pill {
    display: inline-block;
    background: #1a2e1a;
    color: #86efac;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.error-pill {
    display: inline-block;
    background: #2d1515;
    color: #fca5a5;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Final report card */
.report-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1.2rem;
}

/* Guardrail message */
.guardrail-msg {
    background: #1c1408;
    border: 1px solid #92400e;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #fcd34d;
    font-size: 0.88rem;
    margin-top: 1rem;
}
.clarify-msg {
    background: #0f1f2e;
    border: 1px solid #1e40af;
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #93c5fd;
    font-size: 0.88rem;
    margin-top: 1rem;
}

/* Example buttons */
.stButton > button {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.4rem 0.8rem;
    transition: all 0.15s;
    width: 100%;
    text-align: left;
}
.stButton > button:hover {
    background: #253347;
    border-color: #7dd3fc;
    color: #7dd3fc;
}

/* Text area */
.stTextArea textarea {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

/* Divider */
hr {
    border-color: #1e293b !important;
    margin: 1.2rem 0 !important;
}

/* Streamlit label overrides */
label, .stTextArea label {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ── Preset examples ────────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the latest advancements in retrieval-augmented generation for production systems?",
    "Compare LangGraph and AutoGen as agent frameworks — strengths, weaknesses, and use cases.",
    "How do transformer attention mechanisms scale with sequence length, and what are the best solutions?",
]

# ── Session state init ─────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "stream_log" not in st.session_state:
    st.session_state.stream_log = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "clarification" not in st.session_state:
    st.session_state.clarification = None
if "running" not in st.session_state:
    st.session_state.running = False


def reset_state():
    st.session_state.stream_log = []
    st.session_state.final_report = None
    st.session_state.error_message = None
    st.session_state.clarification = None


def format_stream_log(log: list[dict]) -> str:
    """Render the stream log as HTML for the stream panel."""
    if not log:
        return '<span style="color:#334155;font-style:italic;">Waiting for agent to start…</span>'

    lines = []
    for entry in log:
        ts = entry.get("ts", "")
        kind = entry.get("kind", "")
        msg = entry.get("msg", "")

        if kind == "node":
            lines.append(
                f'<div style="margin:4px 0;">'
                f'<span style="color:#475569">[{ts}]</span> '
                f'<span class="node-pill">NODE</span>'
                f'<span style="color:#e2e8f0">{msg}</span>'
                f'</div>'
            )
        elif kind == "tool_call":
            lines.append(
                f'<div style="margin:4px 0;">'
                f'<span style="color:#475569">[{ts}]</span> '
                f'<span class="tool-pill">TOOL</span>'
                f'<span style="color:#86efac">{msg}</span>'
                f'</div>'
            )
        elif kind == "tool_result":
            lines.append(
                f'<div style="margin:4px 0 4px 18px;">'
                f'<span style="color:#334155">↳ </span>'
                f'<span style="color:#64748b">{msg}</span>'
                f'</div>'
            )
        elif kind == "error":
            lines.append(
                f'<div style="margin:4px 0;">'
                f'<span style="color:#475569">[{ts}]</span> '
                f'<span class="error-pill">ERROR</span>'
                f'<span style="color:#fca5a5">{msg}</span>'
                f'</div>'
            )
        elif kind == "info":
            lines.append(
                f'<div style="margin:4px 0;">'
                f'<span style="color:#475569">[{ts}]</span> '
                f'<span style="color:#7dd3fc">ℹ </span>'
                f'<span style="color:#94a3b8">{msg}</span>'
                f'</div>'
            )
        elif kind == "done":
            lines.append(
                f'<div style="margin:8px 0 2px 0;padding-top:6px;border-top:1px solid #1e293b;">'
                f'<span style="color:#86efac;font-weight:600">✓ {msg}</span>'
                f'</div>'
            )

    return "\n".join(lines)


def run_agent(query: str, stream_placeholder, report_placeholder):
    """Import and run the agent, streaming events to the UI."""
    ts = lambda: datetime.now().strftime("%H:%M:%S")

    def log(kind, msg):
        st.session_state.stream_log.append({"ts": ts(), "kind": kind, "msg": msg})
        stream_placeholder.markdown(
            f'<div class="stream-panel">{format_stream_log(st.session_state.stream_log)}</div>',
            unsafe_allow_html=True,
        )
    try:
        # Import the agent graph — adjust this import to match your actual module
        from agent import build_graph
        graph = build_graph()  # type: ignore

        log("info", f'Query received: "{query[:80]}{"…" if len(query) > 80 else ""}"')
        log("node", " input_validator — validating query…")

        initial_state = {
            "query": query,
            "search_results": [],
            "scraped_content": [],
            "rag_results": [],
            "tool_errors": [],
            "iterations": 0,
            "total_tokens": 0,
            "final_report": None,
            "error_message": None,
            "needs_clarification": False,
            "clarification_question": None,
            "cost_usd": 0.0,
        }

        final_state = None

        # Stream events from LangGraph
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_output is None:
                    node_output = {}

                if node_name == "input_validator":
                    if node_output.get("error_message"):
                        log("error", f'Invalid input: {node_output["error_message"]}')
                        st.session_state.error_message = node_output["error_message"]
                        return
                    elif node_output.get("needs_clarification"):
                        log("info", "Query needs clarification")
                        st.session_state.clarification = node_output.get(
                            "clarification_question", "Could you provide more details?"
                        )
                        return
                    else:
                        log("info", "✓ Input valid, proceeding to planner")

                elif node_name == "planner":
                    log("node", " planner — deciding next action…")
                    tool = node_output.get("next_tool")
                    if tool:
                        log("info", f"Planner selected tool: {tool}")

                elif node_name == "tool_caller":
                    log("node", " tool_caller — executing tool…")
                    tool_name = node_output.get("last_tool_called", "unknown tool")
                    log("tool_call", f" {tool_name}(…)")

                    # Summarise what came back
                    sr = node_output.get("search_results", [])
                    sc = node_output.get("scraped_content", [])
                    rr = node_output.get("rag_results", [])
                    errs = node_output.get("tool_errors", [])

                    if sr:
                        log("tool_result", f"{len(sr)} search result(s) retrieved")
                    if sc:
                        log("tool_result", f"{len(sc)} page(s) scraped")
                    if rr:
                        log("tool_result", f"{len(rr)} RAG result(s) returned")
                    if errs:
                        latest_err = errs[-1] if errs else ""
                        log("error", f"Tool error: {latest_err}")

                    iters = node_output.get("iterations", 0)
                    cost = node_output.get("cost_usd", 0.0)
                    log("info", f"Iteration {iters} | Estimated cost: ${cost:.4f}")

                elif node_name == "synthesizer":
                    log("node", " synthesizer — writing final report…")

                final_state = node_output  # keep last known state

        if final_state:
            report = final_state.get("final_report") or (
                # fall back: pull from any accumulated state
                st.session_state.get("_last_report")
            )
            if report:
                log("done", "Agent finished — report ready")
                st.session_state.final_report = report
            else:
                log("error", "Synthesizer returned no report")

    except Exception as e:
        import traceback
        log("error", f"Real error: {traceback.format_exc()}")
        log("info", "⚠ falling back to demo mode")
        _demo_run(query, log)


def _demo_run(query: str, log):
    """Simulate an agent run for UI demonstration when agent.py isn't importable."""
    import time

    steps = [
        ("node", " input_validator — validating query…"),
        ("info", "✓ Input valid"),
        ("node", " planner — deciding next action…"),
        ("info", "Planner selected tool: web_search"),
        ("node", " tool_caller — executing web_search…"),
        ("tool_call", " web_search(query='…')"),
        ("tool_result", "5 search results retrieved"),
        ("info", "Iteration 1 | Estimated cost: $0.0012"),
        ("node", " planner — deciding next action…"),
        ("info", "Planner selected tool: scrape_page"),
        ("node", " tool_caller — executing scrape_page…"),
        ("tool_call", " scrape_page(url='https://example.com/…')"),
        ("tool_result", "1 page scraped (4 231 chars)"),
        ("info", "Iteration 2 | Estimated cost: $0.0023"),
        ("node", " planner — stopping (3 results collected)"),
        ("node", " synthesizer — writing final report…"),
        ("done", "Agent finished — report ready"),
    ]

    for kind, msg in steps:
        log(kind, msg)
        time.sleep(0.25)

    st.session_state.final_report = f"""## Research Report

**Query:** {query}

---

### Overview

This is a **demo report** generated because `agent.py` is not importable in the current environment. In production, the synthesizer node calls the Groq LLM to produce a structured markdown report with citations.

### Key Findings

1. **Finding one** — sourced from web search results and scraped pages.
2. **Finding two** — cross-referenced against the Task 1 RAG knowledge base.
3. **Finding three** — synthesised across multiple sources for completeness.

### Sources

| # | Source | Type |
|---|--------|------|
| 1 | https://example.com/article-1 | Web search |
| 2 | https://example.com/article-2 | Scraped page |
| 3 | RAG corpus chunk #42 | Internal RAG |

### Limitations

- Demo mode: no live data was fetched.
- Production run would include real citations and a cost breakdown.

---
*Report generated in demo mode — connect agent.py to see a live run.*
"""


# ── Layout ─────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="agent-sub">LangGraph · Groq · DuckDuckGo · RAG</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.6], gap="large")

# ── LEFT: Input ────────────────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">Research Question</div>', unsafe_allow_html=True)

    query_input = st.text_area(
        label="query",
        value=st.session_state.query,
        height=130,
        placeholder="e.g. How does contrastive decoding improve LLM output quality?",
        label_visibility="collapsed",
    )

    run_btn = st.button("▶  Run Agent", use_container_width=True, type="primary")

    st.markdown('<div class="section-label" style="margin-top:1rem;">Load Example</div>', unsafe_allow_html=True)

    for i, ex in enumerate(EXAMPLES):
        label = f"Ex {i+1}  {ex[:55]}…" if len(ex) > 55 else f"Ex {i+1}  {ex}"
        if st.button(label, key=f"ex_{i}"):
            st.session_state.query = ex
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Run Info</div>', unsafe_allow_html=True)
    if st.session_state.stream_log:
        total = len([e for e in st.session_state.stream_log if e["kind"] in ("node", "tool_call")])
        st.markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#64748b;">'
            f'{total} events logged</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#334155;">No run yet</span>',
            unsafe_allow_html=True,
        )

# ── RIGHT: Stream panel ────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">Agent Stream</div>', unsafe_allow_html=True)
    stream_placeholder = st.empty()
    stream_placeholder.markdown(
        f'<div class="stream-panel">{format_stream_log(st.session_state.stream_log)}</div>',
        unsafe_allow_html=True,
    )

# ── Report / guardrail output (full width below) ───────────────────────────────
report_placeholder = st.empty()

if st.session_state.error_message:
    report_placeholder.markdown(
        f'<div class="guardrail-msg">'
        f'<strong>⚠ Invalid Input</strong><br>{st.session_state.error_message}'
        f'</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.clarification:
    report_placeholder.markdown(
        f'<div class="clarify-msg">'
        f'<strong>🤔 Clarification Needed</strong><br>{st.session_state.clarification}'
        f'</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.final_report:
    with report_placeholder.container():
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Final Report</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-card">{st.session_state.final_report}</div>',
            unsafe_allow_html=True,
        )

# ── Run logic ──────────────────────────────────────────────────────────────────
if run_btn:
    q = query_input.strip()
    if not q:
        st.warning("Please enter a research question before running.")
    else:
        reset_state()
        st.session_state.query = q
        with st.spinner("Agent running…"):
            run_agent(q, stream_placeholder, report_placeholder)
        st.rerun()