import os
import json
import time

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from groq import Groq

# ── RAGAS + LangChain imports ────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
# ─────────────────────────────────────────────────────────────


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL   = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_ANS_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") 
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "multi_docs"
TOP_K            = 10
TOP_K_RERANKED   = 5
RESULTS_FILE     = "results.json"


# ============================================================
# STEP 1 — CONFIGURE RAGAS TO USE GROQ (not OpenAI)
# ============================================================
def configure_ragas():
    """
    By default RAGAS calls OpenAI. This function points it at
    Groq (Llama 3.1-8b) and uses a local HuggingFace embedding
    model so no OpenAI key is needed at all.
    """
    print("Configuring RAGAS to use Groq...")

    groq_llm = LangchainLLMWrapper(ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
    ))

    hf_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    ))

    # Point every metric at Groq + local embeddings
    faithfulness.llm            = groq_llm
    answer_relevancy.llm        = groq_llm
    answer_relevancy.embeddings = hf_embeddings
    context_precision.llm       = groq_llm

    print("  RAGAS configured — using Groq + HuggingFace embeddings")


# ============================================================
# STEP 2 — LOAD RESOURCES
# ============================================================
def load_resources():
    print("\nLoading ChromaDB...")
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
# STEP 3 — PIPELINE FUNCTIONS
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
    system = """You are a search query optimizer for a documentation assistant.
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
    if not chunks:
        return "I could not find any relevant information in the documentation.", []

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    system = """You are a helpful documentation assistant.
Answer questions using ONLY the provided documentation chunks.
After each sentence add citation numbers like [1] or [2].
If the answer is not in the chunks say: I could not find this in the provided documentation.
Never make up information. Be clear and concise."""

    user = f"""Question: {question}

Documentation chunks:
{context}

Answer using only the chunks above. Add [1][2] citations."""

    resp = groq_client.chat.completions.create(
        model=GROQ_ANS_MODEL,
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
    rewritten      = rewrite_query(groq_client, question)
    hybrid_results = hybrid_search(
        collection, bm25_index, all_chunks,
        all_metadatas, all_ids, rewritten, top_k=TOP_K
    )
    reranked       = rerank(reranker, question, hybrid_results, top_k=TOP_K_RERANKED)
    answer, used   = generate_answer(groq_client, question, reranked)
    contexts       = [chunk["text"] for chunk in used]
    sources        = [chunk["source"] for chunk in used]
    return answer, contexts, sources, rewritten


# ============================================================
# STEP 4 — LOAD QUESTIONS
# ============================================================
def load_questions(filepath="questions.json"):
    print(f"\nLoading questions from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"  Loaded {len(questions)} questions")
    return questions


# ============================================================
# STEP 5 — RUN ALL QUESTIONS THROUGH THE PIPELINE
# ============================================================
def collect_results(questions, collection, bm25_index, all_chunks,
                    all_metadatas, all_ids, reranker, groq_client):
    ragas_questions     = []
    ragas_answers       = []
    ragas_contexts      = []
    ragas_ground_truths = []
    out_of_scope_results = []
    all_results          = []

    print(f"\n{'='*55}")
    print(f"Running pipeline on {len(questions)} questions...")
    print(f"{'='*55}")

    for i, q in enumerate(questions, start=1):
        question_text   = q["question"]
        expected_answer = q["expected_answer"]
        category        = q["category"]

        print(f"\n[{i}/{len(questions)}] {category.upper()}: {question_text[:60]}...")

        try:
            answer, contexts, sources, rewritten = run_pipeline(
                question_text, collection, bm25_index, all_chunks,
                all_metadatas, all_ids, reranker, groq_client
            )

            print(f"  Rewritten : {rewritten[:50]}...")
            print(f"  Sources   : {', '.join(sources[:3])}")
            print(f"  Answer    : {answer[:80]}...")

            if category == "out_of_scope":
                refused = (
                    "could not find" in answer.lower() or
                    "not in" in answer.lower() or
                    "don't have" in answer.lower() or
                    len(contexts) == 0
                )
                out_of_scope_results.append({
                    "question":          question_text,
                    "correctly_refused": refused,
                    "answer":            answer
                })
                print(f"  Refusal test: {'✅ PASSED' if refused else '❌ FAILED'}")
                all_results.append({
                    "id":                q["id"],
                    "category":          category,
                    "question":          question_text,
                    "expected_answer":   expected_answer,
                    "generated_answer":  answer,
                    "sources":           sources,
                    "rewritten_query":   rewritten,
                    "correctly_refused": refused,
                    "included_in_ragas": False
                })

            else:
                ragas_questions.append(question_text)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts)
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

        except Exception as e:
            print(f"  ERROR on question {i}: {e}")
            all_results.append({
                "id":       q["id"],
                "category": category,
                "question": question_text,
                "error":    str(e)
            })

        # Respect Groq free-tier rate limit (~30 req/min)
        time.sleep(2)

    return (ragas_questions, ragas_answers, ragas_contexts,
            ragas_ground_truths, out_of_scope_results, all_results)


# ============================================================
# STEP 6 — RUN RAGAS
# ============================================================
def run_ragas(questions, answers, contexts, ground_truths):
    print(f"\n{'='*55}")
    print(f"Running RAGAS on {len(questions)} in-scope questions...")
    print(f"{'='*55}")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    print("Dataset built. Calculating scores (this will take ~15 mins due to rate limiting)...")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        raise_exceptions=False,
    )

    return result


# ============================================================
# STEP 7 — PRINT AND SAVE RESULTS
# ============================================================
def _mean_metric(ragas_result, metric_name: str) -> float:
    """Average a RAGAS metric from EvaluationResult (ragas >= 0.2)."""
    df = ragas_result.to_pandas()
    if metric_name not in df.columns:
        return 0.0
    return round(float(df[metric_name].mean(skipna=True)), 4)


def print_scores(ragas_result, out_of_scope_results):
    print(f"\n{'='*55}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*55}")

    faithfulness_score      = _mean_metric(ragas_result, "faithfulness")
    answer_relevancy_score  = _mean_metric(ragas_result, "answer_relevancy")
    context_precision_score = _mean_metric(ragas_result, "context_precision")
    average = round(
        (faithfulness_score + answer_relevancy_score + context_precision_score) / 3, 4
    )

    print(f"\n  Faithfulness      : {faithfulness_score}")
    print(f"  Answer Relevancy  : {answer_relevancy_score}")
    print(f"  Context Precision : {context_precision_score}")
    print(f"  {'─'*35}")
    print(f"  Average Score     : {average}")
    print(f"\n  Score Guide:")
    print(f"  0.8–1.0 = Excellent  |  0.6–0.8 = Good")
    print(f"  0.4–0.6 = Acceptable |  <0.4 = Needs work")

    print(f"\n{'='*55}")
    print("OUT-OF-SCOPE REFUSAL TEST")
    print(f"{'='*55}")
    passed = sum(1 for r in out_of_scope_results if r["correctly_refused"])
    total  = len(out_of_scope_results)
    print(f"\n  Correctly refused: {passed}/{total}")
    for r in out_of_scope_results:
        status = "✅ PASSED" if r["correctly_refused"] else "❌ FAILED"
        print(f"\n  {status}: {r['question']}")
        print(f"  Answer: {r['answer'][:120]}...")

    return {
        "faithfulness":      faithfulness_score,
        "answer_relevancy":  answer_relevancy_score,
        "context_precision": context_precision_score,
        "average":           average,
        "refusal_passed":    f"{passed}/{total}"
    }


def save_results(all_results, scores, filepath=RESULTS_FILE):
    output = {
        "ragas_scores":     scores,
        "total_questions":  len(all_results),
        "detailed_results": all_results
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {filepath}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("RAG Documentation Assistant — RAGAS Evaluation")
    print("=" * 55)

    # Step 1: Point RAGAS at Groq (no OpenAI key needed)
    configure_ragas()

    # Step 2: Load all pipeline resources
    (collection, all_chunks, all_metadatas,
     all_ids, bm25_index, reranker, groq_client) = load_resources()

    # Step 3: Load questions
    questions = load_questions("questions.json")

    # Step 4: Run pipeline on all questions
    (ragas_questions, ragas_answers, ragas_contexts,
     ragas_ground_truths, out_of_scope_results,
     all_results) = collect_results(
        questions, collection, bm25_index, all_chunks,
        all_metadatas, all_ids, reranker, groq_client
    )

    print(f"\n{'='*55}")
    print(f"  In-scope questions  : {len(ragas_questions)}")
    print(f"  Out-of-scope tested : {len(out_of_scope_results)}")
    print(f"{'='*55}")

    # Step 5: Run RAGAS
    ragas_result = run_ragas(
        ragas_questions,
        ragas_answers,
        ragas_contexts,
        ragas_ground_truths
    )

    # Step 6: Print scores
    scores = print_scores(ragas_result, out_of_scope_results)

    # Step 7: Save to results.json
    save_results(all_results, scores)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)
    print(f"  Average RAGAS Score : {scores['average']}")
    print(f"  Refusal Test        : {scores['refusal_passed']}")
    print(f"  Results saved to    : {RESULTS_FILE}")


if __name__ == "__main__":
    main()