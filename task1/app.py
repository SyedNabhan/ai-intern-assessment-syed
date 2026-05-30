# Streamlit UI — Multi-Framework RAG Documentation Assistant
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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "multi_docs"
TOP_K           = 10
TOP_K_RERANKED  = 5

# Refusal phrases — if the LLM says any of these, treat it as a refusal
REFUSAL_PHRASES = [
    "could not find",
    "not in the documentation",
    "not covered in",
    "no information",
    "outside the scope",
    "not mentioned",
    "i don't have",
    "i do not have",
    "cannot find",
    "not available in",
]

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Docs Assistant",
    page_icon="logo.png",
    layout="wide",
)


# ============================================================
# LOAD RESOURCES — cached so models load only once
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

    tokenized  = [c.lower().split() for c in all_chunks]
    bm25_index = BM25Okapi(tokenized)
    reranker   = CrossEncoder(RERANKER_MODEL)
    groq_client = Groq(api_key=GROQ_API_KEY)

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
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]
    ids        = results["ids"][0]
    formatted  = []
    for i in range(len(documents)):
        formatted.append({
            "id":       ids[i],
            "text":     documents[i],
            "source":   metadatas[i].get("filename", "unknown"),
            "filepath": metadatas[i].get("source", "unknown"),
            "score":    round(1 - distances[i], 4),
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
            "score":    round(float(scores[idx]), 4),
        })
    return formatted


def hybrid_search(collection, bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    vector_results = vector_search(collection, question, top_k=top_k)
    bm25_results   = bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=top_k)

    bm25_scores    = [r["score"] for r in bm25_results]
    max_bm25       = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_score_map = {r["id"]: round(r["score"] / max_bm25, 4) for r in bm25_results}

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
            "bm25_score":   b,
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


# ============================================================
# QUERY REWRITING — handles ALL question types
# ============================================================
def rewrite_query(groq_client, question, chat_history=None):
    """
    Rewrites any question into a standalone search query.

    Handles:
    - Vague follow-ups ("what about it?", "give me an example")
    - Pronouns ("how does it work?", "can they be combined?")
    - Short/ambiguous questions ("more?", "elaborate")
    - Clear questions (returned as-is)
    - Greetings and chit-chat (returned as-is so pipeline can refuse them)
    """

    # Build last 4 user turns as context
    history_text = ""
    if chat_history:
        recent = [m for m in chat_history if m["role"] == "user"][-4:]
        history_text = "\n".join(f"User: {m['content']}" for m in recent)

    prompt = f"""You are a search query rewriting assistant for a documentation chatbot.

Your job: convert the LATEST user question into a fully standalone search query
that can be understood WITHOUT any conversation context.

Rules:
1. Resolve all pronouns (it, they, this, that, them) using conversation history.
2. If the question is a follow-up like "give me an example" or "elaborate",
   expand it into a full question using the previous topic.
3. If the question is already clear and standalone, return it exactly as-is.
4. If the question is a greeting ("hi", "hello", "thanks") or chit-chat,
   return it exactly as-is — do NOT rewrite it.
5. Keep the rewritten query under 20 words.
6. Return ONLY the rewritten query. No explanation. No punctuation changes.

Examples:
  History: "User: What is FastAPI?"
  Latest: "How do I install it?"
  Output: How do I install FastAPI?

  History: "User: Explain dependency injection in FastAPI."
  Latest: "Give me an example"
  Output: Give me an example of dependency injection in FastAPI

  History: "User: What are LangChain retrievers?"
  Latest: "How do they work with vector stores?"
  Output: How do LangChain retrievers work with vector stores?

  Latest: "What is a path parameter?"  (no history needed)
  Output: What is a path parameter?

  Latest: "hi"
  Output: hi

Conversation history:
{history_text if history_text else "(no history)"}

Latest question: {question}

Rewritten query:"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        # Safety: if rewrite comes back empty or too long, fall back
        if not rewritten or len(rewritten.split()) > 30:
            return question
        return rewritten
    except Exception:
        return question


# ============================================================
# ANSWER GENERATION — handles all edge cases
# ============================================================
def generate_answer(groq_client, question, chunks, chat_history=None):
    """
    Generates a grounded answer from retrieved chunks.

    Handles:
    - Normal factual questions
    - Multi-chunk synthesis questions
    - Out-of-scope questions (refuses cleanly)
    - Greetings and chit-chat (responds naturally)
    - Vague questions (asks a focused follow-up)
    - Code questions (returns properly formatted code)
    """

    # --- Greeting / chit-chat detection ---
    greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good evening"]
    if question.strip().lower() in greetings or len(question.strip().split()) <= 2:
        # Handle short social messages without hitting the docs
        small_talk_responses = {
            "hi": "Hi! Ask me anything about FastAPI or LangChain documentation.",
            "hello": "Hello! What would you like to know about FastAPI or LangChain?",
            "hey": "Hey! Feel free to ask about FastAPI or LangChain.",
            "thanks": "You're welcome! Let me know if you have more questions.",
            "thank you": "Happy to help! Anything else about FastAPI or LangChain?",
            "bye": "Goodbye! Come back anytime with more questions.",
        }
        for key, val in small_talk_responses.items():
            if key in question.strip().lower():
                return val, []
        return "Hi! Ask me anything about FastAPI or LangChain documentation.", []

    # --- No chunks at all ---
    if not chunks:
        return (
            "I could not find any relevant information in the documentation for your question. "
            "Try rephrasing or asking about a specific FastAPI or LangChain concept.",
            []
        )

    # --- Build conversation context for multi-turn awareness ---
    conversation_context = ""
    if chat_history:
        recent = [m for m in chat_history if m["role"] in ("user", "assistant")][-6:]
        lines  = []
        for m in recent:
            role    = "User" if m["role"] == "user" else "Assistant"
            content = m["content"][:300]  # truncate long answers
            lines.append(f"{role}: {content}")
        conversation_context = "\n".join(lines)

    # --- Build numbered chunk context ---
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    system = """You are a precise documentation assistant for FastAPI and LangChain.

STRICT RULES — follow every one:

1. Answer using ONLY the provided documentation chunks.
   Never use outside knowledge or training data.

2. If the answer spans multiple chunks, combine them into one clear answer.

3. Cite every claim with [1], [2], etc. matching the chunk numbers.

4. For code questions, always include a code block using triple backticks.

5. If the chunks clearly do not contain the answer, respond with exactly:
   "I could not find this in the provided documentation."
   Do NOT guess, infer, or make up an answer.

6. If the question is very vague and you need clarification,
   answer what you can from the chunks, then ask ONE focused follow-up.
   Do NOT ask follow-ups for clear questions.

7. Keep answers concise and well-structured.
   Use bullet points for lists of steps or features.
   Use prose for explanations.

8. Never repeat the question back to the user."""

    user = f"""Conversation so far:
{conversation_context if conversation_context else "(this is the first message)"}

Current question: {question}

Documentation chunks:
{context}

Answer the question using only the chunks. Cite with [1][2] etc."""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, chunks

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}", []


# ============================================================
# FULL PIPELINE
# ============================================================
def run_full_pipeline(question, mode, chat_history,
                      collection, bm25_index,
                      all_chunks, all_metadatas,
                      all_ids, reranker, groq_client):

    # Step 1: Rewrite query for better retrieval
    rewritten = rewrite_query(groq_client, question, chat_history)

    # Step 2: Retrieve
    if mode == "Hybrid + Reranker":
        results = hybrid_search(collection, bm25_index, all_chunks, all_metadatas, all_ids, rewritten)
        results = rerank(reranker, question, results)
    else:
        results = vector_search(collection, rewritten, top_k=TOP_K_RERANKED)
        for r in results:
            r["reranker_score"]   = r["score"]
            r["pre_rerank_score"] = r["score"]

    # Step 3: Generate answer with full conversation context
    answer, used_chunks = generate_answer(
        groq_client, question, results, chat_history
    )

    return answer, used_chunks, rewritten


# ============================================================
# HELPERS — detect refusal for UI styling
# ============================================================
def is_refusal(answer):
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


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

        if rewritten_query:
            st.caption(f"Searched as: *{rewritten_query}*")
        st.divider()

        if not used_chunks:
            st.warning("No relevant chunks found.")
            return

        for i, chunk in enumerate(used_chunks, start=1):
            score = chunk.get("reranker_score", chunk.get("score", 0))

            if score > 3:
                badge = "🟢"
            elif score > 0:
                badge = "🟡"
            else:
                badge = "🔴"

            with st.expander(f"[{i}] {chunk['source']}  {badge} {score}"):
                st.caption(f"File: `{chunk['source']}`")
                st.caption(f"Reranker score: `{score}`")
                if "vector_score" in chunk:
                    st.caption(f"Vector: `{chunk['vector_score']}`  |  BM25: `{chunk['bm25_score']}`")
                st.divider()
                st.text(chunk["text"][:400])


# ============================================================
# ANSWER DISPLAY — with citations and refusal handling
# ============================================================
def render_answer_with_citations(answer, used_chunks):
    # Out-of-scope / refusal
    if is_refusal(answer) or not used_chunks:
        st.warning(f"⚠️ {answer}")
        return

    # Normal answer
    st.markdown(answer)

    # Sources list
    if used_chunks:
        st.divider()
        st.caption("**Sources used:**")
        seen = set()
        for i, chunk in enumerate(used_chunks, start=1):
            source = chunk["source"]
            score  = chunk.get("reranker_score", chunk.get("score", "N/A"))
            if source not in seen:
                st.caption(f"`[{i}]` **{source}** — score: {score}")
                seen.add(source)


# ============================================================
# MAIN APP
# ============================================================
def main():

    # Header
    try:
        st.image("logo.png", width=80)
    except Exception:
        pass  # logo is optional — don't crash if missing

    st.title("Multi-Framework Documentation Assistant")
    st.caption("Ask questions about FastAPI and LangChain documentation.")
    st.divider()

    # Load resources
    with st.spinner("Loading models and database... (first load takes ~30 seconds)"):
        try:
            (collection, all_chunks, all_metadatas,
             all_ids, bm25_index, reranker, groq_client) = load_all_resources()
        except Exception as e:
            st.error(f"Failed to load resources: {e}")
            st.stop()

    # Mode toggle
    mode = st.radio(
        "**Retrieval mode**",
        options=["Hybrid + Reranker", "Vector Only"],
        horizontal=True,
        help="Hybrid + Reranker: BM25 + vector + reranking (best quality). Vector Only: faster, basic semantic search."
    )

    if mode == "Hybrid + Reranker":
        st.info("✅ Full pipeline: Query rewriting → Hybrid search → Reranker → Answer")
    else:
        st.info("⚡ Basic pipeline: Query rewriting → Vector search → Answer")

    st.divider()

    # Chat history init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_answer_with_citations(msg["content"], msg.get("chunks", []))

    # Render sidebar for last assistant message
    last_assistant = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        None
    )
    if last_assistant:
        render_sidebar(
            last_assistant.get("chunks", []),
            mode,
            last_assistant.get("rewritten", "")
        )

    # Chat input
    question = st.chat_input("Ask a question about FastAPI or LangChain...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role":    "user",
            "content": question
        })

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                try:
                    answer, used_chunks, rewritten_query = run_full_pipeline(
                        question, mode,
                        st.session_state.messages,
                        collection, bm25_index,
                        all_chunks, all_metadatas, all_ids,
                        reranker, groq_client
                    )
                except Exception as e:
                    answer        = f"Something went wrong: {str(e)}"
                    used_chunks   = []
                    rewritten_query = question

            render_answer_with_citations(answer, used_chunks)

        # Update sidebar
        render_sidebar(used_chunks, mode, rewritten_query)

        # Save to history
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   answer,
            "chunks":    used_chunks,
            "rewritten": rewritten_query
        })

    # Empty state — shown before first question
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("How do I create a POST route in FastAPI?")
        with col2:
            st.info("How can FastAPI and LangChain be combined to build RAG apps?")
        with col3:
            st.info("How does LangChain support Retrieval-Augmented Generation?")


if __name__ == "__main__":
    main()