
import os
import json
import time
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from groq import Groq

# RAGAS imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
# faithfulness     — does the answer stick to the retrieved chunks?
# answer_relevancy — does the answer actually answer the question?
# context_precision — were the retrieved chunks relevant to the question?

from datasets import Dataset
# Dataset is a HuggingFace library that RAGAS needs to format the data


# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL     = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH     = "./chroma_db"
COLLECTION_NAME    = "fastapi_docs"
TOP_K              = 10
TOP_K_RERANKED     = 5

# Output file where results are saved
RESULTS_FILE       = "results.json"


# ============================================================
# STEP 1 — LOAD ALL RESOURCES
# ============================================================
def load_resources():
    """
    Loads ChromaDB, BM25 index, reranker, and Groq client.
    Same as in app.py but without Streamlit caching.
    """
    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    print(f"  ChromaDB loaded — {collection.count()} chunks")

    print("Loading all chunks for BM25...")
    all_data      = collection.get(include=["documents", "metadatas"])
    all_chunks    = all_data["documents"]
    all_metadatas = all_data["metadatas"]
    all_ids       = all_data["ids"]
    tokenized     = [c.lower().split() for c in all_chunks]
    bm25_index    = BM25Okapi(tokenized)
    print(f"  BM25 index built — {len(all_chunks)} chunks")

    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("  Reranker loaded")

    print("Loading Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("  Groq loaded")

    return collection, all_chunks, all_metadatas, all_ids, bm25_index, reranker, groq_client


# ============================================================
# STEP 2 — PIPELINE FUNCTIONS (same as retrieval.py)
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
            "id":     ids[i],
            "text":   documents[i],
            "source": metadatas[i].get("filename", "unknown"),
            "score":  similarities[i]
        })
    return formatted


def bm25_search(bm25_index, all_chunks, all_metadatas, all_ids, question, top_k=TOP_K):
    tokenized_q = question.lower().split()
    scores      = bm25_index.get_scores(tokenized_q)
    top_indices = np.argsort(scores)[::-1][:top_k]
    formatted   = []
    for idx in top_indices:
        formatted.append({
            "id":     all_ids[idx],
            "text":   all_chunks[idx],
            "source": all_metadatas[idx].get("filename", "unknown"),
            "score":  round(float(scores[idx]), 4)
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
        v = vector_score_map.get(cid, 0.0)
        b = bm25_score_map.get(cid, 0.0)
        chunk = all_chunks_map[cid]
        hybrid_results.append({
            "id":     cid,
            "text":   chunk["text"],
            "source": chunk["source"],
            "score":  round(0.5 * v + 0.5 * b, 4)
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_results[:top_k]


def rerank(reranker, question, results, top_k=TOP_K_RERANKED):
    pairs  = [[question, r["text"]] for r in results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(results):
        r["reranker_score"] = round(float(scores[i]), 4)
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
    """Generate answer with citations. Returns answer string and used chunks."""
    if not chunks:
        return "I could not find any relevant information in the FastAPI documentation.", []

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    system = """You are a helpful FastAPI documentation assistant.
Answer questions using ONLY the provided documentation chunks.
After each sentence add citation numbers like [1] or [2].
If the answer is not in the chunks say: I could not find this in the provided documentation.
Never make up information. Be clear and concise."""

    user = f"""Question: {question}

Documentation chunks:
{context}

Answer using only the chunks above. Add [1][2] citations."""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip(), chunks


def run_pipeline(question, collection, bm25_index, all_chunks,
                 all_metadatas, all_ids, reranker, groq_client):
    """
    Runs the full pipeline for one question.
    Returns: answer string, list of context strings, list of source filenames
    """
    # Step 1: Rewrite query
    rewritten = rewrite_query(groq_client, question)

    # Step 2: Hybrid search
    hybrid_results = hybrid_search(
        collection, bm25_index, all_chunks,
        all_metadatas, all_ids, rewritten, top_k=TOP_K
    )

    # Step 3: Rerank
    reranked = rerank(reranker, question, hybrid_results, top_k=TOP_K_RERANKED)

    # Step 4: Generate answer
    answer, used_chunks = generate_answer(groq_client, question, reranked)

    # Extract just the text from each chunk for RAGAS
    # RAGAS needs contexts as a list of strings — not dicts
    contexts = [chunk["text"] for chunk in used_chunks]
    sources  = [chunk["source"] for chunk in used_chunks]

    return answer, contexts, sources, rewritten


# ============================================================
# STEP 3 — READ QUESTIONS FROM questions.json
# ============================================================
def load_questions(filepath="questions.json"):
    """
    Reads the questions.json file and returns the list of questions.
    """
    print(f"\nLoading questions from {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    print(f"  Loaded {len(questions)} questions")
    return questions


# ============================================================
# STEP 4 — RUN ALL QUESTIONS THROUGH THE PIPELINE
# ============================================================
def collect_results(questions, collection, bm25_index, all_chunks,
                    all_metadatas, all_ids, reranker, groq_client):
    """
    Loops through all 25 questions, runs each through the pipeline,
    and collects results in the format RAGAS expects.

    RAGAS needs 4 lists of equal length:
    - questions:   the original questions
    - answers:     your system's generated answers
    - contexts:    list of retrieved chunk texts for each question
    - ground_truths: the expected answers from questions.json

    Out-of-scope questions (expected_answer = "NOT_IN_DOCS") are
    handled separately — we check if the system correctly refused.
    """

    # These lists will be passed to RAGAS
    ragas_questions    = []  # questions to evaluate (in-scope only)
    ragas_answers      = []  # your system's answers
    ragas_contexts     = []  # retrieved chunks for each question
    ragas_ground_truths = [] # expected answers from questions.json

    # These track out-of-scope question performance separately
    out_of_scope_results = []

    # This saves ALL results for your results.json file
    all_results = []

    print(f"\n{'='*55}")
    print(f"Running pipeline on {len(questions)} questions...")
    print(f"{'='*55}")

    for i, q in enumerate(questions, start=1):
        question_text    = q["question"]
        expected_answer  = q["expected_answer"]
        category         = q["category"]

        print(f"\n[{i}/25] {category.upper()}: {question_text[:60]}...")

        try:
            # Run the full pipeline
            answer, contexts, sources, rewritten = run_pipeline(
                question_text, collection, bm25_index, all_chunks,
                all_metadatas, all_ids, reranker, groq_client
            )

            print(f"  Rewritten: {rewritten[:50]}...")
            print(f"  Sources:   {', '.join(sources[:3])}")
            print(f"  Answer:    {answer[:80]}...")

            # ── OUT OF SCOPE HANDLING ────────────────────────
            if category == "out_of_scope":
                # Check if the system correctly refused
                refused = (
                    "could not find" in answer.lower() or
                    "not in" in answer.lower() or
                    len(contexts) == 0
                )
                out_of_scope_results.append({
                    "question": question_text,
                    "correctly_refused": refused,
                    "answer": answer
                })
                print(f"  Refusal test: {'✅ PASSED' if refused else '❌ FAILED — should have refused'}")

                # Save to all_results but skip RAGAS (out-of-scope has no ground truth)
                all_results.append({
                    "id":               q["id"],
                    "category":         category,
                    "question":         question_text,
                    "expected_answer":  expected_answer,
                    "generated_answer": answer,
                    "sources":          sources,
                    "rewritten_query":  rewritten,
                    "correctly_refused": refused,
                    "included_in_ragas": False
                })
                # ────────────────────────────────────────────

            else:
                # ── IN-SCOPE QUESTIONS → add to RAGAS data ──
                ragas_questions.append(question_text)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts)
                # RAGAS needs ground_truths as a list of strings
                ragas_ground_truths.append(expected_answer)

                all_results.append({
                    "id":               q["id"],
                    "category":         category,
                    "question":         question_text,
                    "expected_answer":  expected_answer,
                    "generated_answer": answer,
                    "sources":          sources,
                    "rewritten_query":  rewritten,
                    "included_in_ragas": True
                })
                # ────────────────────────────────────────────

        except Exception as e:
            # If one question fails, don't crash the whole evaluation
            print(f"  ERROR on question {i}: {e}")
            all_results.append({
                "id":       q["id"],
                "category": category,
                "question": question_text,
                "error":    str(e)
            })

        # Wait 1 second between questions to avoid Groq rate limiting
        # Groq free tier allows ~30 requests per minute
        time.sleep(1)

    return (ragas_questions, ragas_answers, ragas_contexts,
            ragas_ground_truths, out_of_scope_results, all_results)


# ============================================================
# STEP 5 — RUN RAGAS
# ============================================================
def run_ragas(questions, answers, contexts, ground_truths):
    """
    Runs RAGAS evaluation on the collected results.

    RAGAS needs a HuggingFace Dataset with exactly these columns:
    - question:     the question asked
    - answer:       your system's answer
    - contexts:     list of retrieved chunk texts
    - ground_truth: the expected correct answer

    Returns a dictionary of metric scores.
    """
    print(f"\n{'='*55}")
    print("Running RAGAS evaluation...")
    print(f"Evaluating {len(questions)} in-scope questions")
    print(f"{'='*55}")

    # Build the dataset RAGAS expects
    # Each key maps to a list — one entry per question
    data = {
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        # contexts must be a list of lists:
        # [ ["chunk1 text", "chunk2 text"], ["chunk1 text", ...], ... ]
        "ground_truth": ground_truths
    }

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_dict(data)
    print("Dataset created successfully")

    # Run RAGAS with the 3 metrics
    # This calls your LLM (via OpenAI by default) to evaluate
    # We configure it to use Groq instead
    print("Calculating scores... (this takes 2-5 minutes)")
    print("RAGAS makes one API call per question per metric")

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
    )

    return result


# ============================================================
# STEP 6 — PRINT AND SAVE RESULTS
# ============================================================
def print_scores(ragas_result, out_of_scope_results):
    """
    Prints RAGAS scores in a readable format and
    shows out-of-scope refusal performance.
    """
    print(f"\n{'='*55}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*55}")

    # Extract scores — RAGAS returns them as a dict
    scores = ragas_result

    faithfulness_score     = round(float(scores["faithfulness"]),     4)
    answer_relevancy_score = round(float(scores["answer_relevancy"]), 4)
    context_precision_score = round(float(scores["context_precision"]), 4)

    # Calculate average
    average = round(
        (faithfulness_score + answer_relevancy_score + context_precision_score) / 3, 4
    )

    print(f"\n  Faithfulness      : {faithfulness_score}")
    print(f"  Answer Relevancy  : {answer_relevancy_score}")
    print(f"  Context Precision : {context_precision_score}")
    print(f"  {'─'*35}")
    print(f"  Average Score     : {average}")

    # Score interpretation
    print(f"\n  Score Guide:")
    print(f"  0.8 - 1.0 = Excellent")
    print(f"  0.6 - 0.8 = Good")
    print(f"  0.4 - 0.6 = Acceptable")
    print(f"  Below 0.4 = Needs improvement")

    # Out-of-scope results
    print(f"\n{'='*55}")
    print("OUT-OF-SCOPE REFUSAL TEST")
    print(f"{'='*55}")

    passed = sum(1 for r in out_of_scope_results if r["correctly_refused"])
    total  = len(out_of_scope_results)
    print(f"\n  Correctly refused: {passed}/{total}")

    for r in out_of_scope_results:
        status = "✅ PASSED" if r["correctly_refused"] else "❌ FAILED"
        print(f"\n  {status}: {r['question']}")
        print(f"  Answer: {r['answer'][:100]}...")

    print(f"\n{'='*55}")

    return {
        "faithfulness":      faithfulness_score,
        "answer_relevancy":  answer_relevancy_score,
        "context_precision": context_precision_score,
        "average":           average,
        "refusal_passed":    f"{passed}/{total}"
    }


def save_results(all_results, scores, filepath=RESULTS_FILE):
    """
    Saves all results and scores to a JSON file.
    This is your proof of evaluation — include it in your submission.
    """
    output = {
        "ragas_scores": scores,
        "total_questions": len(all_results),
        "detailed_results": all_results
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {filepath}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("FastAPI RAG — RAGAS Evaluation (Day 4)")
    print("=" * 55)

    # ── IMPORTANT NOTE ABOUT RAGAS API ──────────────────────
    # RAGAS by default uses OpenAI to evaluate your answers.
    # If you don't have an OpenAI key, set these env variables
    # to use a different LLM for evaluation.
    # The simplest approach: set OPENAI_API_KEY to your Groq key
    # and override the base URL. OR just add an OpenAI free trial.
    #
    # Easiest fix: add this to your .env file:
    # OPENAI_API_KEY=your_groq_key_here
    #
    # RAGAS will use it for the evaluation LLM calls.
    # ────────────────────────────────────────────────────────

    # Step 1: Load resources
    (collection, all_chunks, all_metadatas,
     all_ids, bm25_index, reranker, groq_client) = load_resources()

    # Step 2: Load questions
    questions = load_questions("questions.json")

    # Step 3: Run all questions through pipeline
    (ragas_questions, ragas_answers, ragas_contexts,
     ragas_ground_truths, out_of_scope_results,
     all_results) = collect_results(
        questions, collection, bm25_index, all_chunks,
        all_metadatas, all_ids, reranker, groq_client
    )

    print(f"\n{'='*55}")
    print(f"Pipeline complete!")
    print(f"  In-scope questions processed : {len(ragas_questions)}")
    print(f"  Out-of-scope questions tested: {len(out_of_scope_results)}")
    print(f"{'='*55}")

    # Step 4: Run RAGAS
    ragas_result = run_ragas(
        ragas_questions,
        ragas_answers,
        ragas_contexts,
        ragas_ground_truths
    )

    # Step 5: Print scores
    scores = print_scores(ragas_result, out_of_scope_results)

    # Step 6: Save results
    save_results(all_results, scores)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE!")
    print("=" * 55)
    print(f"  RAGAS Average Score : {scores['average']}")
    print(f"  Refusal Test        : {scores['refusal_passed']}")
    print(f"\n  Results saved to    : {RESULTS_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Look at the 3 lowest-scoring questions in results.json")
    print(f"  2. Fix them by tweaking chunk size, TOP_K, or the prompt")
    print(f"  3. Run evaluate.py again and show before/after scores in README")


if __name__ == "__main__":
    main()
