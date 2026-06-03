# decisions.md

## Chunking Strategy

Used **Recursive Character Chunking** with a chunk size of **500 tokens** and an overlap of **50 tokens**.

FastAPI and LangChain docs are written in medium-length Markdown sections. A chunk size of 500 keeps enough context for meaningful answers without diluting relevance. The 50-token overlap prevents answers from being cut off at chunk boundaries.

## Embedding Model

Used **`sentence-transformers/all-MiniLM-L6-v2`**.

It's fast, lightweight, and runs locally without any API cost. It performs well on technical documentation and is a standard choice for semantic search in RAG systems.

## Vector Database

Used **ChromaDB** over Qdrant for simplicity — no server setup required, works locally out of the box, and is well-suited for prototyping.

## Retrieval Strategy

Combined **BM25 (keyword)** and **semantic vector search** into a hybrid retrieval pipeline. BM25 handles exact technical terms (e.g. function names, class names) while semantic search handles meaning-based queries. Results are merged and reranked using a **CrossEncoder (`ms-marco-MiniLM-L-6-v2`)** for final ordering.

## LLM

Used **Llama 3.3 70B** via the **Groq API** for answer generation. Groq's inference speed makes it ideal for a responsive chat UI, and Llama 3.3 70B handles technical documentation Q&A well with accurate citation-grounded responses.

## Corpus

Ingested documentation from two sources:

- **FastAPI** — official docs from `github.com/tiangolo/fastapi` (`/docs/en/docs` folder), covering routing, dependency injection, authentication, request handling, and more.
- **LangChain** — official docs covering chains, retrievers, embeddings, agents, memory, and document loaders.

Both corpora are in Markdown format and were parsed, chunked, embedded, and stored in ChromaDB as separate collections.
