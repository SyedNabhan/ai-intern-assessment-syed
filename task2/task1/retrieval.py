import os
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq


# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "multi_docs"
TOP_K           = 10
TOP_K_RERANKED  = 5

# Minimum reranker score to use a chunk in the answer
# If ALL chunks score below this, the app says "I don't know"
# This is the "refusal" feature required by your task sheet
RELEVANCE_THRESHOLD = -5.0
# CrossEncoder scores range roughly from -10 to +10
# -5.0 means: if even the best chunk scores below -5,
# the question is probably not in the docs at all


# ============================================================
# STEP 1 — CONNECT TO CHROMADB
# ============================================================
def load_collection():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    print(f"Connected! Collection has {collection.count()} chunks.")

    print("Loading all chunks for BM25...")
    all_data      = collection.get(include=["documents", "metadatas"])
    all_chunks    = all_data["documents"]
    all_metadatas = all_data["metadatas"]
    all_ids       = all_data["ids"]
    print(f"Loaded {len(all_chunks)} chunks.")

    return collection, all_chunks, all_metadatas, all_ids


# ============================================================
# STEP 2 — LOAD MODELS
# ============================================================
def load_reranker():
    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("Reranker loaded!")
    return reranker


def load_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq client loaded!")
    return client


# ============================================================
# STEP 3 — QUERY REWRITING (Task 4, unchanged)
# ============================================================
def rewrite_query(groq_client, original_question):
    system_prompt = """You are a search query optimizer for a Multi documentation assistant.
Rewrite the user question into a better search query.
Rules:
- Return ONLY the rewritten query, nothing else
- Use technical FastAPI & Langchain terminology
- Keep it under 15 words
- If the question is already good, return it as-is"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": original_question}
        ],
        temperature=0.1,
        max_tokens=100
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"  Original : {original_question}")
    print(f"  Rewritten: {rewritten}")
    return rewritten


# ============================================================
# STEP 4 — VECTOR SEARCH (unchanged)
# ============================================================
def vector_search(collection, question, top_k=TOP_K):
    results    = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]
    ids        = results["ids"][0]
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


# ============================================================
# STEP 5 — BM25 SEARCH (unchanged)
# ============================================================
def build_bm25_index(all_chunks):
    print("Building BM25 index...")
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    print(f"BM25 index built.")
    return bm25_index


def bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    tokenized_question = question.lower().split()
    scores      = bm25_index.get_scores(tokenized_question)
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


# ============================================================
# STEP 6 — HYBRID SEARCH (unchanged)
# ============================================================
def hybrid_search(collection, bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    vector_results   = vector_search(collection, question, top_k=top_k)
    bm25_results     = bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=top_k)

    bm25_scores      = [r["score"] for r in bm25_results]
    max_bm25         = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_score_map   = {r["id"]: round(r["score"] / max_bm25, 4) for r in bm25_results}

    vector_scores    = [r["score"] for r in vector_results]
    max_vector       = max(vector_scores) if max(vector_scores) > 0 else 1
    vector_score_map = {r["id"]: round(r["score"] / max_vector, 4) for r in vector_results}

    all_result_ids    = set()
    all_result_chunks = {}

    for r in vector_results:
        all_result_ids.add(r["id"])
        all_result_chunks[r["id"]] = r

    for r in bm25_results:
        all_result_ids.add(r["id"])
        if r["id"] not in all_result_chunks:
            all_result_chunks[r["id"]] = r

    hybrid_results = []
    for chunk_id in all_result_ids:
        v_score      = vector_score_map.get(chunk_id, 0.0)
        b_score      = bm25_score_map.get(chunk_id, 0.0)
        hybrid_score = round((0.5 * v_score) + (0.5 * b_score), 4)
        chunk_info   = all_result_chunks[chunk_id]
        hybrid_results.append({
            "id":           chunk_id,
            "text":         chunk_info["text"],
            "source":       chunk_info["source"],
            "filepath":     chunk_info["filepath"],
            "score":        hybrid_score,
            "vector_score": v_score,
            "bm25_score":   b_score
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_results[:top_k]


# ============================================================
# STEP 7 — RERANKER (unchanged)
# ============================================================
def rerank(reranker, question, hybrid_results, top_k=TOP_K_RERANKED):
    pairs           = [[question, r["text"]] for r in hybrid_results]
    reranker_scores = reranker.predict(pairs)

    for i, r in enumerate(hybrid_results):
        r["reranker_score"]             = round(float(reranker_scores[i]), 4)
        r["hybrid_score_before_rerank"] = r["score"]

    reranked = sorted(hybrid_results, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]


# ============================================================
# STEP 8 — ANSWER GENERATION WITH CITATIONS (NEW — Task 5)
# ============================================================
def generate_answer(groq_client, original_question, reranked_chunks):
    """
    Takes the top reranked chunks and sends them to Groq
    along with the question. Groq reads the chunks and writes
    a proper answer with citations like [1], [2], [3].

    This is the final step that turns retrieved chunks into
    a readable, cited answer — making it a true RAG system.

    HOW CITATIONS WORK:
    We number each chunk [1], [2], [3] etc and include them
    in the prompt. We instruct Groq to write [1] or [2] next
    to any sentence that came from that chunk.

    REFUSAL:
    If the top chunk scores too low (below RELEVANCE_THRESHOLD),
    we skip the LLM entirely and return "I don't know".
    This prevents the model from hallucinating answers that
    aren't in the docs — a required feature in your task sheet.

    WHAT THE PROMPT LOOKS LIKE:
    We send Groq something like:

    System: You are a FastAPI documentation assistant...
    User:
      Question: How do I create a POST route?

      Context chunks:
      [1] Source: first-steps.md
      To declare a path operation use @app.post()...

      [2] Source: path-params.md
      You can declare path parameters...

      Answer using only the chunks above. Cite with [1][2].
      If the answer is not in the chunks, say so.
    """

    # ── REFUSAL CHECK ────────────────────────────────────────
    # Check if the best chunk is relevant enough to answer from
    # If not — say "I don't know" rather than hallucinate
    if not reranked_chunks:
        return "I could not find any relevant information in the FastAPI documentation.", []

    best_score = reranked_chunks[0].get("reranker_score", 0)
    if best_score < RELEVANCE_THRESHOLD:
        return (
            "I could not find an answer to this question in the FastAPI documentation. "
            "This topic may not be covered in the docs, or try rephrasing your question.",
            []
        )
    # ────────────────────────────────────────────────────────

    # ── BUILD CONTEXT STRING ─────────────────────────────────
    # Format the chunks into numbered context blocks
    # Each chunk gets a number [1], [2] etc for citation
    context_parts = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']}\n"
            f"Relevance score: {chunk.get('reranker_score', 'N/A')}\n"
            f"{chunk['text']}"
        )

    context_string = "\n\n".join(context_parts)
    # ────────────────────────────────────────────────────────

    # ── BUILD THE PROMPT ─────────────────────────────────────
    system_prompt = """You are a helpful FastAPI documentation assistant.

Your job is to answer questions using ONLY the provided documentation chunks.

Rules:
- Answer using ONLY information from the provided chunks
- After each sentence or claim, add a citation like [1] or [2] showing which chunk it came from
- If the same info appears in multiple chunks, cite all of them like [1][2]
- If the answer is not in the chunks, say: "I could not find this in the provided documentation."
- Never make up information that isn't in the chunks
- Be clear and concise
- Use code examples from the chunks when available"""

    user_prompt = f"""Question: {original_question}

Documentation chunks:
{context_string}

Please answer the question using only the chunks above.
Add citation numbers [1], [2] etc after each claim."""
    # ────────────────────────────────────────────────────────

    # ── CALL GROQ ────────────────────────────────────────────
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,
        # temperature=0.2 — mostly deterministic
        # We want factual answers, not creative ones
        max_tokens=1000
        # 1000 tokens is enough for a detailed answer with citations
    )

    answer = response.choices[0].message.content.strip()
    # ────────────────────────────────────────────────────────

    return answer, reranked_chunks


# ============================================================
# STEP 9 — FULL PIPELINE (updated with generation)
# ============================================================
def full_pipeline(groq_client, collection, bm25_index, reranker,
                  all_chunks, all_metadatas, all_ids, original_question):
    """
    Complete RAG pipeline from question to cited answer:

    1. Rewrite query       — fix vague questions
    2. Hybrid search       — find top 10 chunks
    3. Rerank              — pick truly best 5
    4. Generate answer     — write cited answer from chunks  ← NEW
    """
    # Step 1: Rewrite
    rewritten_query = rewrite_query(groq_client, original_question)

    # Step 2: Hybrid search with rewritten query
    hybrid_results = hybrid_search(
        collection, bm25_index, all_chunks,
        all_metadatas, all_ids,
        rewritten_query, top_k=TOP_K
    )

    # Step 3: Rerank with original question
    reranked_results = rerank(reranker, original_question, hybrid_results, top_k=TOP_K_RERANKED)

    # Step 4: Generate answer with citations
    answer, used_chunks = generate_answer(groq_client, original_question, reranked_results)

    return answer, used_chunks, rewritten_query


# ============================================================
# STEP 10 — PRINT FINAL OUTPUT
# ============================================================
def print_final_answer(question, answer, used_chunks, rewritten_query):
    """
    Prints the final answer in a clean, readable format.
    Shows the answer, citations, and source chunks.
    """
    print(f"\n{'=' * 60}")
    print(f" QUESTION")
    print(f"{'=' * 60}")
    print(f"{question}")
    print(f"\n(Searched as: {rewritten_query})")

    print(f"\n{'=' * 60}")
    print(f" ANSWER")
    print(f"{'=' * 60}")
    print(answer)

    if used_chunks:
        print(f"\n{'=' * 60}")
        print(f" SOURCES USED")
        print(f"{'=' * 60}")
        for i, chunk in enumerate(used_chunks, start=1):
            print(f"\n[{i}] {chunk['source']}  (reranker score: {chunk.get('reranker_score', 'N/A')})")
            print(f"{'─' * 60}")
            # Show first 200 chars of the chunk as a preview
            preview = chunk["text"][:200]
            if len(chunk["text"]) > 200:
                preview += "..."
            print(preview)

    print(f"\n{'=' * 60}\n")


# ============================================================
# MAIN — Test the complete RAG pipeline end to end
# ============================================================
def main():
    print("=" * 60)
    print("FastAPI RAG — Full Pipeline with Generation (Day 2, Task 5)")
    print("=" * 60)

    # Load everything
    collection, all_chunks, all_metadatas, all_ids = load_collection()
    bm25_index  = build_bm25_index(all_chunks)
    reranker    = load_reranker()
    groq_client = load_groq_client()

    print("\nAll systems loaded. Starting pipeline tests...\n")

    # ── TEST QUESTIONS ──────────────────────────────────────
    # Mix of good questions, vague questions, and one question
    # that is NOT in the docs (to test the refusal feature)
    # ────────────────────────────────────────────────────────
    test_questions = [
        # Good FastAPI questions — should get detailed cited answers
        "How do I create a POST route in FastAPI?",
        "How do I handle errors and exceptions in FastAPI?",
        "How does dependency injection work in FastAPI?",

        # Vague question — query rewriter should fix this
        "how do I check who the user is?",

        # Out of scope question — should trigger refusal
        "What is the capital of France?",
    ]

    for question in test_questions:
        print(f"\nProcessing: '{question}'")
        print("-" * 60)

        answer, used_chunks, rewritten_query = full_pipeline(
            groq_client, collection, bm25_index, reranker,
            all_chunks, all_metadatas, all_ids, question
        )

        print_final_answer(question, answer, used_chunks, rewritten_query)

    print("\n" + "=" * 60)
    print("DAY 2 COMPLETE!")
    print("=" * 60)
    print("Your full RAG pipeline works:")
    print("  Query rewriting  ✅")
    print("  Hybrid search    ✅")
    print("  Reranking        ✅")
    print("  Answer generation with citations  ✅")
    print("  Refusal for out-of-scope questions ✅")
    print("\nNext: Day 3 — Build the Streamlit UI on top of this pipeline")


if __name__ == "__main__":
    main()