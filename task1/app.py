# ============================================================
# app.py — FastAPI Documentation RAG — Streamlit Web App
# ============================================================
# Day 3 — All 5 tasks in one file:
#   Task 1 ✅ — Basic chat input and answer display
#   Task 2 ✅ — Inline citations
#   Task 3 ✅ — Side panel with chunks and scores
#   Task 4 ✅ — Comparison toggle (vector-only vs hybrid+reranker)
#   Task 5 ✅ — Refusal message for out-of-scope questions
#
# To run: streamlit run app.py
# ============================================================

import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

load_dotenv()

EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL     = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH     = "./chroma_db"
COLLECTION_NAME    = "multi_docs"
TOP_K              = 10
TOP_K_RERANKED     = 5


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi Docs Assistant",
    page_icon="logo.png",
    layout="wide",
)


# ============================================================
# LOAD ALL MODELS — cached so they only load ONCE
# ============================================================
@st.cache_resource
def load_all_resources():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    all_data      = collection.get(include=["documents", "metadatas"])
    all_chunks    = all_data["documents"]
    all_metadatas = all_data["metadatas"]
    all_ids       = all_data["ids"]
    tokenized     = [c.lower().split() for c in all_chunks]
    bm25_index    = BM25Okapi(tokenized)
    reranker      = CrossEncoder(RERANKER_MODEL)
    groq_client   = Groq(api_key=GROQ_API_KEY)
    return collection, all_chunks, all_metadatas, all_ids, bm25_index, reranker, groq_client


# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================
def vector_search(collection, question, top_k=TOP_K):
    results      = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    documents    = results["documents"][0]
    metadatas    = results["metadatas"][0]
    distances    = results["distances"][0]
    ids          = results["ids"][0]
    similarities = [round(1 - d, 4) for d in distances]
    formatted    = []
    for i in range(len(documents)):
        formatted.append({
            "id":       ids[i],
            "text":     documents[i],
            "source":   metadatas[i].get("filename", "unknown"),
            "filepath": metadatas[i].get("source", "unknown"),
            "score":    similarities[i]

    
        })
    return formatted


def bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    tokenized_q = question.lower().split()
    scores      = bm25_index.get_scores(tokenized_q)
    top_indices = np.argsort(scores)[::-1][:top_k]
    formatted   = []
    for idx in top_indices:
        formatted.append({
            "id":       all_ids[idx],
            "text":     all_chunks[idx],
            "source":   all_metadatas[idx].get("filename", "unknown"),
            "filepath": all_metadatas[idx].get("source", "unknown"),
            "score":    round(float(scores[idx]), 4)
        })
    return formatted


def hybrid_search(collection, bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    vector_results = vector_search(collection, question, top_k=top_k)
    bm25_results   = bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=top_k)

    bm25_scores      = [r["score"] for r in bm25_results]
    max_bm25         = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_score_map   = {r["id"]: round(r["score"] / max_bm25, 4) for r in bm25_results}

    vector_scores    = [r["score"] for r in vector_results]
    max_vector       = max(vector_scores) if max(vector_scores) > 0 else 1
    vector_score_map = {r["id"]: round(r["score"] / max_vector, 4) for r in vector_results}

    all_ids_set    = set()
    all_chunks_map = {}
    for r in vector_results:
        all_ids_set.add(r["id"])
        all_chunks_map[r["id"]] = r
    for r in bm25_results:
        all_ids_set.add(r["id"])
        if r["id"] not in all_chunks_map:
            all_chunks_map[r["id"]] = r

    hybrid_results = []
    for cid in all_ids_set:
        v     = vector_score_map.get(cid, 0.0)
        b     = bm25_score_map.get(cid, 0.0)
        chunk = all_chunks_map[cid]
        hybrid_results.append({
            "id":           cid,
            "text":         chunk["text"],
            "source":       chunk["source"],
            "filepath":     chunk["filepath"],
            "score":        round(0.5 * v + 0.5 * b, 4),
            "vector_score": v,
            "bm25_score":   b
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_results[:top_k]


def rerank(reranker, question, results, top_k=TOP_K_RERANKED):
    pairs  = [[question, r["text"]] for r in results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(results):
        r["reranker_score"]   = round(float(scores[i]), 4)
        r["pre_rerank_score"] = r["score"]
    reranked = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]


def rewrite_query(groq_client, question):
    system = """You are a FastAPI and LangChain documentation expert.
You MUST answer the question using the documentation chunks provided.
ALWAYS cite with [1][2] after each sentence showing which chunk supports it.
NEVER say you cannot find something if chunks are provided to you.
Use the chunks even if they only partially answer the question.
Be detailed, clear and helpful.

Rewrite the user question into a concise search query.

Rules:
- Return keywords only.
- Do NOT answer the question.
- Do NOT explain.
- Do NOT write complete sentences.
- Maximum 10 words.

Examples:

What is FastAPI?
→ FastAPI framework

How do I install FastAPI?
→ FastAPI installation

What are linked lists?
→ linked lists

How does OAuth2 work?
→ OAuth2 authentication

IMPORTANT FORMATTING RULES:
Only show code when it is absolutely necessary or when user asks explicitly.
- When showing ANY code, ALWAYS wrap it in triple backticks with the language name like this:
```python
your code here
```
- Every code example must be in its own separate code block
- Never write code inline in a sentence — always use a code block
- If the question involves implementation, always include at least one working code example"""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question}
        ],
        temperature=0.1,
        max_tokens=100
    )
    return resp.choices[0].message.content.strip()

def is_relevant_question(groq_client, question, chat_history=[]):

    q_lower = question.lower().strip()
    q_words = q_lower.split()

    # ── ALWAYS BLOCK these regardless of history ──────────────
    blocked_starters = [
        "who is", "who was", "who are", "who were",
        "what is the capital", "when was", "where is",
        "how old is", "tell me about the person",
    ]

    for blocked in blocked_starters:
        if q_lower.startswith(blocked):
            return False

    # ── ALWAYS ALLOW if chat history exists and question
    #    is a short follow-up (under 8 words) ──────────────────
    # "how to implement it", "give me the code", "show example",
    # "what does that mean", "can you explain" etc
    if chat_history and len(q_words) <= 8:
        return True

    # ── ALWAYS ALLOW common follow-up phrases ─────────────────
    followup_phrases = [
        "how to implement", "give me the code", "show me the code",
        "show example", "full code", "complete example",
        "explain more", "elaborate", "expand", "continue",
        "what does that mean", "can you explain", "more detail",
        "how do i do", "what about", "and how", "also how",
        "implement it", "code for it", "example for it",
        "show it", "demonstrate", "give code"
    ]
    if chat_history and any(phrase in q_lower for phrase in followup_phrases):
        return True

    # ── LLM CHECK for longer standalone questions ──────────────
    system = """You are a strict relevance checker for a documentation assistant
that covers ONLY FastAPI and LangChain.

Reply with ONLY one word: YES or NO.

Say YES ONLY if the question is directly about:
- FastAPI: routes, middleware, Pydantic, dependencies, auth, OAuth2, JWT, CORS, websockets
- LangChain: agents, chains, tools, LLMs, memory, embeddings, retrievers, prompts, RAG
- Code implementation using FastAPI or LangChain

Say Strictly NO for absolutely everything else:
- People, celebrities, athletes (Messi, Nolan, anyone)
- Geography, capitals, cities, countries
- General CS: linked lists, sorting, recursion, algorithms
- Math, science, history, sports, cooking, news
- Questions about any specific person
- Questions about computer science terms that are completely irrelevant to the documents ingested (corpus)
- Questions that are irrelevant to the corpus such as variables , linked lists , operators and similar.
- Questions about Business and Celebrities.

    blocked_topics = [
        "linked list", "linked lists", "binary tree", "binary search",
        "bubble sort", "merge sort", "quick sort", "sorting algorithm",
        "data structure", "stack and queue", "hash table", "hash map",
        "recursion", "dynamic programming", "big o notation",
        "capital of", "population of", "who invented",
        "lionel messi", "cristiano ronaldo", "christopher nolan",
        "football", "soccer", "basketball", "cricket",
        "what is python", "what is javascript", "what is java",
        "what is c++", "what is html", "what is css" And any of irrelevant Computer topics
    ]

When in doubt say NO.
"""


    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question}
        ],
        temperature=0.0,
        max_tokens=5
    )
    answer = resp.choices[0].message.content.strip().upper()
    return answer.startswith("YES")

def generate_answer(groq_client, question, chunks):
    """
    Generate answer from chunks. Always answers if chunks exist.
    Only refuses if truly no chunks were retrieved at all.
    """
    # Only refuse if there are truly no chunks
    if not chunks:
        return "I could not find any relevant information in the FastAPI documentation.", []

    # Build numbered context
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    system = """You are a FastAPI documentation expert.
You MUST answer the question using the documentation chunks provided.
ALWAYS cite with [1][2] after each sentence showing which chunk supports it.
NEVER say you cannot find something if chunks are provided to you.
Use the chunks even if they only partially answer the question.
Be detailed, clear and helpful."""

    user = f"""Question: {question}

Use these documentation chunks to answer:
{context}

Give a detailed answer with [1][2] citations after each sentence."""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    answer = resp.choices[0].message.content.strip()
    return answer, chunks

def is_irrelevant_answer(answer):
    """
    Detects when the LLM generated an answer that admits
    the topic is not in the documentation.
    Returns True if the answer should be replaced with a refusal.
    """
    irrelevant_phrases = [
        "not explicitly mentioned in the provided documentation",
        "not mentioned in the provided documentation",
        "not explicitly mentioned in the provided",
        "does not provide a direct",
        "not provide a direct definition",
        "not provide any information about",
        "not specifically mentioned",
        "not directly mentioned",
        "cannot find this in the provided",
        "not found in the documentation",
        "documentation does not mention",
        "chunks do not include",
        "chunks do not provide",
        "not covered in the documentation",
        "no information about",
        "outside the scope",
        "not related to fastapi",
        "not related to langchain",
    ]

    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in irrelevant_phrases)


def run_full_pipeline(question, mode, collection, bm25_index,
                      all_chunks, all_metadatas, all_ids, reranker,
                      groq_client, chat_history=[]):

    # Step 1: Check relevance — pass chat history so follow-ups are allowed
    if not is_relevant_question(groq_client, question, chat_history):
        refusal = (
            "⚠️ I can only answer questions about **FastAPI** and **LangChain** documentation.\n\n"
            "Please ask something related to these topics — for example:\n"
            "- How do I create a POST route in FastAPI?\n"
            "- How does dependency injection work in FastAPI?\n"
            "- How do I build an agent with tools in LangChain?\n"
            "- What are LangChain chains and how do I use them?"
        )
        return refusal, [], "off-topic — question refused"

    # Step 2: Rewrite query — include last assistant answer for context
    if chat_history:
        last_exchange = chat_history[-1] if chat_history else ""
        contextual_question = f"Previous context: {last_exchange}\n\nCurrent question: {question}"
    else:
        contextual_question = question

    rewritten = rewrite_query(groq_client, contextual_question)

    # Step 3: Retrieve chunks
    if mode == "Hybrid + Reranker":
        results = hybrid_search(
            collection, bm25_index, all_chunks,
            all_metadatas, all_ids, rewritten
        )
        results = rerank(reranker, question, results)
    else:
        results = vector_search(collection, rewritten, top_k=TOP_K_RERANKED)
        for r in results:
            r["reranker_score"]   = r["score"]
            r["pre_rerank_score"] = r["score"]

    # Step 4: Generate answer
    answer, used_chunks = generate_answer(groq_client, question, results)
    return answer, used_chunks, rewritten


# ============================================================
# SIDEBAR — retrieved chunks panel
# ============================================================
def render_sidebar(used_chunks, mode, rewritten_query):
    with st.sidebar:
        st.header("🔍 Retrieved Chunks")

        if mode == "Hybrid + Reranker":
            st.success("Mode: Hybrid + Reranker")
        else:
            st.info("Mode: Vector Only")

        st.caption(f"Searched as: *{rewritten_query}*")
        st.divider()

        if not used_chunks:
            st.warning("No relevant chunks found.")
            return

        for i, chunk in enumerate(used_chunks, start=1):
            score = chunk.get("reranker_score", chunk.get("score", 0))
            score_icon = "🟢" if score > 3 else ("🟡" if score > 0 else "🔴")

            with st.expander(f"[{i}] {chunk['source']}  {score_icon} {score}"):
                st.caption(f"File: `{chunk['source']}`")
                st.caption(f"Reranker score: `{score}`")
                if "vector_score" in chunk:
                    st.caption(f"Vector: `{chunk['vector_score']}`  |  BM25: `{chunk['bm25_score']}`")
                st.divider()
                st.text(chunk["text"][:400])


# ============================================================
# ANSWER DISPLAY — with citations and refusal handling
# ============================================================
def render_answer(answer, used_chunks):
    """
    Displays answer with proper formatting.
    Detects and replaces irrelevant answers with a clean refusal message.
    """
    # Case 1 — No chunks retrieved at all
    if not used_chunks:
        st.warning(answer)
        return

    # Case 2 — LLM admitted the topic is not in the docs
    if is_irrelevant_answer(answer):
        st.warning(
            "❌ **This question is not relevant to the provided documentation.**\n\n"
            "This assistant only answers questions about:\n"
            "- ⚡ **FastAPI** — routes, middleware, auth, dependencies, Pydantic, etc.\n"
            "- 🦜 **LangChain** — agents, chains, tools, LLMs, memory, RAG, etc.\n\n"
            "Please ask a question related to these topics."
        )
        return

    # Case 3 — Normal answer with citations
    st.markdown(answer)
    st.divider()
    st.caption("**Sources used:**")
    for i, chunk in enumerate(used_chunks, start=1):
        score = chunk.get("reranker_score", chunk.get("score", "N/A"))
        st.caption(f"`[{i}]` **{chunk['source']}** — score: {score}")


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================
def display_chat_history():
    """
    Loops through all saved messages and renders them.
    Called at the start of every page load so history persists.
    """
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User message bubble
            with st.chat_message("user"):
                st.markdown(msg["content"])

        elif msg["role"] == "assistant":
            # Assistant message bubble with answer + citations
            with st.chat_message("assistant"):
                render_answer(
                    msg["content"],
                    msg.get("chunks", [])
                )
                # Show the rewritten query as a small caption
                if msg.get("rewritten"):
                    st.caption(f"🔍 Searched as: *{msg['rewritten']}*")


# ============================================================
# MAIN APP
# ============================================================
def main():

    # ── HEADER ───────────────────────────────────────────────
    try:
        st.image("logo.png", width=80)
    except Exception:
        pass  # logo is optional — don't crash if missing

    st.title("Multi-Framework Documentation Assistant")
    st.caption("Ask questions about FastAPI and LangChain documentation.")
    st.divider()

    # ── LOAD RESOURCES ───────────────────────────────────────
    with st.spinner("Loading models... (first load ~30 seconds)"):
        (collection, all_chunks, all_metadatas, all_ids,
         bm25_index, reranker, groq_client) = load_all_resources()

    # ── TASK 4 — COMPARISON TOGGLE ───────────────────────────
    mode = st.radio(
        "**Retrieval mode**",
        options=["Hybrid + Reranker", "Vector Only"],
        horizontal=True,
        help="Hybrid uses BM25 + vector search + reranking. Vector Only uses basic semantic search."
    )

    if mode == "Hybrid + Reranker":
        st.info("✅ Full pipeline: Query rewriting → Hybrid search (vector + BM25) → Reranker → Answer")
    else:
        st.info("⚡ Basic pipeline: Query rewriting → Vector search only → Answer")

    st.divider()

    # ── INITIALISE CHAT HISTORY ──────────────────────────────
    # st.session_state persists data across reruns
    # Every time user submits a question the page reruns
    # Without session_state the history would be lost each time
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── DISPLAY EXISTING CHAT HISTORY ────────────────────────
    # This renders ALL previous messages every time the page loads
    # giving the appearance of a persistent chat
    display_chat_history()

    # ── SHOW SIDEBAR FOR LAST ANSWER ─────────────────────────
    # Find the last assistant message and show its chunks in sidebar
    last_assistant = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_assistant = msg
            break

    if last_assistant:
        render_sidebar(
            last_assistant.get("chunks", []),
            mode,
            last_assistant.get("rewritten", "")
        )

    # ── EMPTY STATE — shown before first question ─────────────
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("How do I create a POST route in FastAPI?")
        with col2:
            st.info("How do retrievers work in LangChain?")
        with col3:
            st.info("How do I handle authentication in FastAPI?")

    # ── CHAT INPUT ───────────────────────────────────────────
    # st.chat_input sits at the bottom of the screen
    # Returns the typed message when user presses Enter
    question = st.chat_input("Ask a question about FastAPI and LangChain...")

    if question:
        # 1. Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # 2. Save user message to history
        st.session_state.messages.append({
            "role":    "user",
            "content": question
        })

        # 3. Run pipeline and show answer
        # Build chat history summary for context
        chat_history_summary = []
        for msg in st.session_state.messages[-4:]:
            # Pass last 4 messages as context (2 exchanges)
            if msg["role"] == "assistant":
                chat_history_summary.append(msg["content"][:200])

        answer, used_chunks, rewritten_query = run_full_pipeline(
            question, mode,
            collection, bm25_index,
            all_chunks, all_metadatas, all_ids,
            reranker, groq_client,
            chat_history=chat_history_summary  # ← pass history here
        )

        # Show answer with citations
        render_answer(answer, used_chunks)
        st.caption(f"🔍 Searched as: *{rewritten_query}*")

        # 4. Update sidebar with this answer's chunks
        render_sidebar(used_chunks, mode, rewritten_query)

        # 5. Save assistant message to history
        # This is what display_chat_history() reads on next page load
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   answer,
            "chunks":    used_chunks,
            "rewritten": rewritten_query
        })

        # 6. Rerun to show updated history cleanly
        st.rerun()


if __name__ == "__main__":
    main()