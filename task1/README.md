# Multi-Framework RAG Documentation Assistant



An AI-powered documentation assistant built using Retrieval-Augmented Generation (RAG) that answers questions about FastAPI and LangChain documentation with grounded responses and source citations.



\## Features



\- FastAPI + LangChain documentation support

\- ChromaDB vector database

\- Hybrid Retrieval (BM25 + Semantic Search)

\- CrossEncoder reranking

\- Citation-based answers

\- Hallucination prevention

\- Streamlit UI

\- RAGAS evaluation



\## Tech Stack



\- Python

\- ChromaDB

\- Sentence Transformers

\- BM25

\- CrossEncoder

\- Streamlit

\- Groq API



\## RAG Pipeline



```text

Docs → Chunking → Embeddings → ChromaDB

→ Hybrid Retrieval → Reranking → LLM → Answer + Citations

```



\## Run Locally



```bash

pip install -r requirements.txt



python ingest.py



streamlit run app.py

```



\## Example Questions



\- What is a retriever in LangChain?

\- How does OAuth2 authentication work in FastAPI?

\- What is hybrid retrieval?

\- How do embeddings improve semantic retrieval?



\## Highlights



\- Built a complete RAG pipeline from scratch

\- Implemented hybrid retrieval + reranking

\- Added grounded citation-based answering

\- Designed hallucination refusal system



\---

Built by Syed Nabhan

BSc Computer Science Student,

The New College 24 - 27
