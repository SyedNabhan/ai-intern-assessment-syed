import sys, os
import sys, os

# Point to task2 and task3 folders
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task2")
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task3")
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task1")
from agent_tools import query_rag, web_search
from agent import build_graph, default_state
from github_client import list_issues, create_issue
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run(question: str, callback=None) -> dict:
    """
    callback(step, msg) — called at each step so UI can stream progress.
    Returns {rag, agent_report, github, final_answer, cost, latency}
    """
    t0 = time.time()
    cost = 0.0

    def log(step, msg):
        if callback:
            callback(step, msg)

    # Step 1 — RAG
    log("rag", "Searching documentation...")
    rag = query_rag(question)
    rag_answer = rag.get("answer", "")
    log("rag", f"RAG: {rag_answer[:100]}..." if rag_answer else "RAG: no answer found")

    # Step 2 — Agent
    log("agent", "Running research agent...")
    graph = build_graph()
    state = default_state(question)
    final = {}
    for event in graph.stream(state, stream_mode="updates"):
        for node, updates in event.items():
            if updates:
                final.update(updates)
                log("agent", f"Agent node: {node}")
    agent_report = final.get("final_report", "")

    # Step 3 — GitHub (Task 3)
    log("github", "Checking GitHub issues...")
    try:
        issues = list_issues(os.getenv("GITHUB_REPO"), limit=3)
        github_context = "\n".join([f"#{i['number']}: {i['title']}" for i in issues])
    except Exception as e:
        github_context = f"GitHub unavailable: {e}"
    log("github", f"GitHub: {github_context[:100]}")

    # Step 4 — Synthesize
    log("synthesizer", "Writing final answer...")
    prompt = f"""You are a developer support assistant.

User question: {question}

Documentation context:
{rag_answer[:1000] if rag_answer else 'No documentation found.'}

Web research:
{agent_report[:1000] if agent_report else 'No web research available.'}

Related GitHub issues:
{github_context}

Write a clear, cited answer. Be concise. End with one suggested follow-up action."""

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    final_answer = resp.choices[0].message.content
    cost += 0.002  # approximate
    latency = round(time.time() - t0, 1)

    return {
        "rag": rag_answer,
        "agent_report": agent_report,
        "github": github_context,
        "final_answer": final_answer,
        "cost": cost,
        "latency": latency,
    }