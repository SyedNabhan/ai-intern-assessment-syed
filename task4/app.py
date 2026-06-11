import sys, os
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task2")
sys.path.insert(0, r"D:\AI_Development\ai-intern-assessment-syed\task3")

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from orchestrator import run
from github_client import create_issue

st.set_page_config(page_title="Dev Support Assistant", layout="wide")
st.title("🛠 Developer Support Assistant")
st.caption("RAG · Agent · GitHub — Task 1 + 2 + 3 integrated")

# ── Input
question = st.chat_input("Ask a question about your project...")

# ── Session state
if "history" not in st.session_state:
    st.session_state.history = []

# ── Chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Run on new question
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        activity = st.status("Running...", expanded=True)
        result_box = st.empty()

        log_lines = []

        def callback(step, msg):
            icons = {"rag": "📚", "agent": "🤖", "github": "🐙", "synthesizer": "✍️"}
            icon = icons.get(step, "•")
            log_lines.append(f"{icon} **{step}** — {msg}")
            activity.write("\n\n".join(log_lines))

        result = run(question, callback=callback)
        activity.update(label="Done", state="complete")

        # Final answer
        st.markdown(result["final_answer"])

        # Cost + latency
        st.caption(f"⏱ {result['latency']}s · 💰 ~${result['cost']:.4f}")

        # Action button — create GitHub issue
        if st.button("📝 Create GitHub issue from this question"):
            from github_client import create_issue
            import os
            r = create_issue(
                os.getenv("GITHUB_REPO"),
                f"Support question: {question[:60]}",
                f"**Question:** {question}\n\n**Answer:**\n{result['final_answer']}",
            )
            st.success(f"Created issue #{r['number']}: {r['url']}")

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": result["final_answer"]})