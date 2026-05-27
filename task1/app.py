import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

# Load environment variables from .env
load_dotenv()

EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL     = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH     = "./chroma_db"
COLLECTION_NAME    = "fastapi_docs"
TOP_K              = 10
TOP_K_RERANKED     = 5


# ============================================================
# PAGE CONFIG — must be the very first Streamlit command
# ============================================================
st.set_page_config(
    page_title="FastAPI Docs Assistant",
    page_icon="⚡",
    layout="wide",
    # wide layout gives us room for the sidebar + main content
)


# ============================================================
# LOAD ALL MODELS — cached so they only load ONCE
# ============================================================
# @st.cache_resource tells Streamlit:
# "Run this function only once. Cache the result.
#  On every page refresh, reuse the cached result."
# Without this, your models would reload on EVERY question — very slow.

@st.cache_resource
def load_all_resources():
    """
    Loads ChromaDB, BM25 index, reranker, and Groq client.
    Called once when the app starts. Cached after that.
    """
    # --- ChromaDB ---
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # --- Load all chunks for BM25 ---
    all_data      = collection.get(include=["documents", "metadatas"])
    all_chunks    = all_data["documents"]
    all_metadatas = all_data["metadatas"]
    all_ids       = all_data["ids"]

    # --- Build BM25 index ---
    tokenized = [c.lower().split() for c in all_chunks]
    bm25_index = BM25Okapi(tokenized)

    # --- Load reranker ---
    reranker = CrossEncoder(RERANKER_MODEL)

    # --- Load Groq ---
    groq_client = Groq(api_key=GROQ_API_KEY)

    return collection, all_chunks, all_metadatas, all_ids, bm25_index, reranker, groq_client


# ============================================================
# RETRIEVAL FUNCTIONS (same logic as retrieval.py)
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

    formatted = []
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

    all_ids_set   = set()
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
        v = vector_score_map.get(cid, 0.0)
        b = bm25_score_map.get(cid, 0.0)
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
        r["reranker_score"] = round(float(scores[i]), 4)
        r["pre_rerank_score"] = r["score"]
    reranked = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]


def rewrite_query(groq_client, question):
    system = """You are a search query optimizer for a FastAPI documentation assistant.
Rewrite the user question into a better search query.
Return ONLY the rewritten query. No explanation. Under 15 words.
If the question is already good, return it as-is."""
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

def generate_answer(groq_client, question, chunks):

    # Only refuse if there are truly no chunks at all
    if not chunks:
        return "I could not find any relevant information in the FastAPI documentation.", []

    # Build context — number each chunk clearly
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    system = """You are a helpful FastAPI documentation assistant.
You MUST answer the question using the provided documentation chunks.
Always cite which chunk you used like [1] or [2] after each sentence.
Even if the chunks only partially answer the question, use what is there.
Never say you cannot find something if the chunks contain relevant content."""

    user = f"""Question: {question}

Here are the documentation chunks to use:
{context}

Answer the question using the chunks above. Cite with [1][2] etc."""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    answer = resp.choices[0].message.content.strip()
    return answer, chunks


def run_full_pipeline(question, mode, collection, bm25_index,
                      all_chunks, all_metadatas, all_ids, reranker, groq_client):
    """
    Runs the pipeline based on the selected mode.
    mode = "Hybrid + Reranker" or "Vector Only"
    """
    # Step 1: Rewrite query
    rewritten = rewrite_query(groq_client, question)

    if mode == "Hybrid + Reranker":
        # Full pipeline
        results  = hybrid_search(collection, bm25_index, all_chunks, all_metadatas, all_ids, rewritten)
        results  = rerank(reranker, question, results)
    else:
        # Vector only — no BM25, no reranker
        results  = vector_search(collection, rewritten, top_k=TOP_K_RERANKED)
        # add dummy reranker_score so display code works uniformly
        for r in results:
            r["reranker_score"]   = r["score"]
            r["pre_rerank_score"] = r["score"]

    # Step 2: Generate answer
    answer, used_chunks = generate_answer(groq_client, question, results)

    return answer, used_chunks, rewritten


# ============================================================
# TASK 3 — SIDEBAR: show retrieved chunks and scores
# ============================================================
def render_sidebar(used_chunks, mode, rewritten_query):
    """
    Renders the sidebar with:
    - Which mode is active
    - The rewritten query
    - Each retrieved chunk with score and source
    """
    with st.sidebar:
        st.header("🔍 Retrieved Chunks")

        # Show active mode
        if mode == "Hybrid + Reranker":
            st.success("Mode: Hybrid + Reranker")
        else:
            st.info("Mode: Vector Only")

        # Show rewritten query
        st.caption(f"Searched as: *{rewritten_query}*")
        st.divider()

        if not used_chunks:
            st.warning("No relevant chunks found.")
            return

        # Show each chunk
        for i, chunk in enumerate(used_chunks, start=1):
            score = chunk.get("reranker_score", chunk.get("score", 0))

            # Color the score — green if high, red if low
            if score > 3:
                score_color = "🟢"
            elif score > 0:
                score_color = "🟡"
            else:
                score_color = "🔴"

            # st.expander creates a collapsible section
            with st.expander(f"[{i}] {chunk['source']}  {score_color} {score}"):
                st.caption(f"File: `{chunk['source']}`")
                st.caption(f"Reranker score: `{score}`")
                if "vector_score" in chunk:
                    st.caption(f"Vector: `{chunk['vector_score']}`  |  BM25: `{chunk['bm25_score']}`")
                st.divider()
                # Show chunk text preview
                st.text(chunk["text"][:400])


# ============================================================
# TASK 2 — CITATIONS: display answer with source list
# ============================================================
def render_answer_with_citations(answer, used_chunks):
    """
    Displays the answer text (which already contains [1][2] markers)
    and then lists the sources below it.
    """
    # TASK 5 — REFUSAL: show warning box for out-of-scope questions
    if not used_chunks or "could not find" in answer.lower():
        st.warning(answer)
        return

    # Display the main answer
    # st.markdown renders the text nicely including [1][2] markers
    st.markdown(answer)

    # Display sources below the answer
    if used_chunks:
        st.divider()
        st.caption("**Sources used:**")
        for i, chunk in enumerate(used_chunks, start=1):
            score = chunk.get("reranker_score", chunk.get("score", "N/A"))
            st.caption(f"`[{i}]` **{chunk['source']}** — score: {score}")


# ============================================================
# MAIN APP — renders the full page
# ============================================================
def main():

    # ── APP HEADER ───────────────────────────────────────────
    st.title("⚡ FastAPI Documentation Assistant")
    st.caption("Ask any question about FastAPI. Answers come with citations from the official docs.")
    st.divider()

    # ── LOAD RESOURCES ───────────────────────────────────────
    # Show a spinner while models load on first run
    with st.spinner("Loading models and database... (first load takes ~30 seconds)"):
        collection, all_chunks, all_metadatas, all_ids, bm25_index, reranker, groq_client = load_all_resources()

    # ── TASK 4 — COMPARISON TOGGLE ───────────────────────────
    # st.radio creates a button group the user can click
    mode = st.radio(
        "**Retrieval mode**",
        options=["Hybrid + Reranker", "Vector Only"],
        horizontal=True,
        # horizontal=True shows buttons side by side instead of stacked
        help="Hybrid + Reranker uses BM25 + vector search + reranking for best results. Vector Only uses basic semantic search."
    )

    # Show what each mode means to the user
    if mode == "Hybrid + Reranker":
        st.info("✅ Full pipeline: Query rewriting → Hybrid search (vector + BM25) → Reranker → Answer")
    else:
        st.info("⚡ Basic pipeline: Query rewriting → Vector search only → Answer")

    st.divider()

    # ── CHAT HISTORY ─────────────────────────────────────────
    # st.session_state is Streamlit's way of remembering data
    # across page refreshes. Without this, chat history disappears
    # every time the user submits a new question.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # messages is a list of dicts:
        # {"role": "user", "content": "..."}
        # {"role": "assistant", "content": "...", "chunks": [...], "rewritten": "..."}

    # Display existing chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_answer_with_citations(msg["content"], msg.get("chunks", []))
                # Also re-render sidebar for the last message
    # ────────────────────────────────────────────────────────

    # ── CHAT INPUT ───────────────────────────────────────────
    # st.chat_input creates the text box at the bottom of the screen
    # It returns the user's message when they press Enter
    question = st.chat_input("Ask a question about FastAPI...")

    if question:
        # Show the user's message in the chat
        with st.chat_message("user"):
            st.markdown(question)

        # Save user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Run the pipeline and show answer
        with st.chat_message("assistant"):
            # Show a spinner while processing
            with st.spinner("Searching documentation and generating answer..."):
                answer, used_chunks, rewritten_query = run_full_pipeline(
                    question, mode,
                    collection, bm25_index,
                    all_chunks, all_metadatas, all_ids,
                    reranker, groq_client
                )

            # TASK 2 — Display answer with citations
            render_answer_with_citations(answer, used_chunks)

        # TASK 3 — Update sidebar with retrieved chunks
        render_sidebar(used_chunks, mode, rewritten_query)

        # Save assistant message to history
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   answer,
            "chunks":    used_chunks,
            "rewritten": rewritten_query
        })

    # ── EMPTY STATE — shown before first question ─────────────
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("How do I create a POST route in FastAPI?")
        with col2:
            st.info("How does dependency injection work?")
        with col3:
            st.info("How do I handle authentication in FastAPI?")


# ── RUN THE APP ──────────────────────────────────────────────
if __name__ == "__main__":
    main()