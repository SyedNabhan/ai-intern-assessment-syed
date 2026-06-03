# Complete Chat Summary — FastAPI RAG Project
## From Zero to Full RAG System in 4 Days

---

## 1. CONTEXT — Who You Are and What You Need

You are a **complete beginner** (your exact words) who has been given a job assessment task. You need to build a RAG (Retrieval-Augmented Generation) system as **Task 1** of a multi-task assessment to get a job. You had **4 days** to complete it. You had GitHub installed but were otherwise starting from scratch with no prior knowledge of RAG, embeddings, vector databases, or Python web frameworks.

---

## 2. THE TASK REQUIREMENTS (exactly as given)

The task was titled **"Task 1 — Documentation Q&A with retrieval"**. The reviewer will pick 3 of their own questions and try them on your app. The app must:
- Answer accurately
- Cite where it found the answer
- Refuse to guess when the answer isn't in the docs

### Required concepts:
- **Chunking** — splitting documents into pieces small enough to embed
- **Embeddings** — converting text into vectors
- **Vector databases** — storing and searching vectors
- **Hybrid retrieval** — combining vector search (semantic) with keyword search (BM25/lexical)
- **Reranking** — using a second model to reorder top results
- **Agentic RAG** — letting the model decide whether to retrieve or answer directly
- **Evaluation with RAGAS** — measuring faithfulness and context precision

### Required features:
- `ingest.py` — parses corpus, chunks, embeds, writes to ChromaDB . Must be idempotent.
- Hybrid retrieval — vector + BM25 with documented scoring
- Reranker — cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Query rewriting before retrieval
- Citations — every claim tied to source file and section
- Streamlit/Gradio UI with chat history, inline citations, side panel showing chunks and scores, toggle to compare two retrieval configurations

### Recommended corpus options:
- PostgreSQL 16 documentation (most challenging)
- Stripe API documentation
- FastAPI documentation

---

## 3. INITIAL SETUP — GitHub and Repository

### The problem
You said you had GitHub installed but didn't know how to use it. You needed step-by-step guidance from scratch.

### What was explained:
- How to create a GitHub account
- How to create a **private repository** named `ai-intern-assessment-<your-name>`
- How to add `.env` to `.gitignore` **before the first commit** — emphasized as critical
- How to share the repo with the assessor's email via Collaborators settings
- How to clone the repo to your Desktop using `git clone`
- How to create task folders with `mkdir task1`
- The basic Git workflow: `git add .` → `git commit -m "message"` → `git push`

### Critical warning given:
"Never commit an API key — even once, even in a deleted commit — is a serious concern."

---

## 4. API KEYS — What You Need and Where to Get Them

### The .env file explained from basics:
- A `.env` file is a plain text file that lives on your computer only
- Never uploaded to GitHub because it's in `.gitignore`
- Stores secrets like API keys safely

### Wrong way (never do this):
```python
api_key = "sk-abc123yoursecretkey"  # hardcoded in code
```

### Right way:
```
# .env file
GROQ_API_KEY=your_key_here
```

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

### The two free services recommended:

**1. Groq** (console.groq.com)
- Free, no credit card needed
- You were asked: Multilingual or Text-to-text? → **Text-to-text**
- You were asked: GPT or Llama? → **Llama**
- You then asked: Llama 4 Scout or Llama 3.3 70B? → **Llama 3.3 70B** (more tested, more stable, tons of tutorials)
- Model string: `llama-3.3-70b-versatile`

**2. Hugging Face** (huggingface.co/settings/tokens)
- Free, token called "Access Token" not "API key"
- Token type: **Read**
- Models download automatically when first called — no dashboard selection needed

### Final `.env` file contents:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Two HuggingFace models explained:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — converts text chunks to numbers
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — reorders search results by true relevance

---

## 5. THE 4-DAY PLAN (structured overview)

### Day 1 — Setup + Read + Ingest (Foundation)
### Day 2 — Retrieval + Generation Pipeline (Core logic)
### Day 3 — Build the UI (Interface)
### Day 4 — Evaluate + Polish + Submit (Final)

---

## 6. CORPUS CHOICE — PostgreSQL vs FastAPI

You initially said you wanted to use **PostgreSQL** (the most challenging corpus). This was discussed:

- PostgreSQL docs are in `.sgml` format (XML-like tags) — needs BeautifulSoup to strip tags
- FastAPI docs are in `.md` format — plain text, easier to read
- You would need `beautifulsoup4` and `lxml` extra libraries for PostgreSQL
- PostgreSQL signals you're serious to reviewers

However, when the actual code was written, you proceeded with **FastAPI** (the `.md` corpus). The final project uses FastAPI documentation.

---

## 7. DAY 1 — DETAILED WALKTHROUGH

### What ingest.py does (explained from scratch):
"Ingest" means to take something in and process it.
1. Reads every `.md` file from the FastAPI docs folder
2. Splits each file into chunks (~500 characters with 50-character overlap)
3. Converts each chunk into an embedding (list of numbers) using HuggingFace
4. Saves all chunks + embeddings into ChromaDB on disk

Analogy given: "Imagine you have a 500-page book. Instead of re-reading it for every question, you build an index first. `ingest.py` builds that index."

### Exact directory structure created:
```
Desktop/
└── ai-intern-assessment-syed/
    ├── .gitignore
    ├── README.md
    └── task1/
        ├── .env
        ├── ingest.py
        └── fastapi/
            └── docs/
                └── en/
                    └── docs/   ← 153 .md files
```

### Libraries installed:
```
pip install langchain
pip install langchain-community
pip install chromadb
pip install sentence-transformers
pip install streamlit
pip install python-dotenv
pip install rank-bm25
pip install groq
pip install ragas
pip install beautifulsoup4
pip install lxml
```

### Issues encountered and fixed:

**Issue 1: Import errors in VS Code**
`Import "chromadb" could not be resolved — Pylance[reportMissingImports]`
- Cause: VS Code using wrong Python interpreter
- Fix: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose correct Python path

**Issue 2: `ModuleNotFoundError: No module named 'langchain.text_splitter'`**
- Cause: Newer langchain moved this module
- Fix: `pip install langchain-text-splitters`
- Code change: `from langchain.text_splitter import RecursiveCharacterTextSplitter` → `from langchain_text_splitters import RecursiveCharacterTextSplitter`

**Issue 3: Script runs but no output appears**
- Long debugging process — tested each import individually, all passed
- Real cause: **The file was not saved properly in VS Code**
- Fix: Used `Ctrl+Shift+S` → named it same → replaced → got output
- Lesson: Always save with `Ctrl+S` before running

**Issue 4: `chromadb.errors.DuplicateIDError`**
- Cause: Multiple files named `index.md` in different folders produced duplicate chunk IDs like `index.md__chunk_0`
- Fix: Changed ID generation to include file number:
  ```python
  # Before:
  ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
  # After:
  ids = [f"{file_num}_{filename}__chunk_{i}" for i in range(len(chunks))]
  ```

### Successful ingest.py output:
```
Found 153 markdown files in fastapi/docs/en/docs
Processing (1/153): alternatives.md → Created 54 chunks
...
Processing (153/153): simple-oauth2.md → Created 23 chunks
Saving 3869 total chunks to database...
  Saved 3869 chunks to ChromaDB
INGESTION COMPLETE!
Total files processed: 153
Total chunks saved: 3869
Database saved at: ./chroma_db
```

### GitHub push issues:
**Error 1**: `fatal: not a git repository`
- Cause: Running `git push` from inside `task1/` instead of the repo root
- Fix: `cd D:\AI_Development\ai-intern-assessment-syed` then push

**Error 2**: `! [rejected] main -> main (non-fast-forward)`
- Cause: GitHub had a README file that local didn't have yet
- Fix: `git pull origin main --allow-unrelated-histories` then `git push origin main`

---

## 8. DAY 2 — RETRIEVAL PIPELINE (5 Tasks)

All 5 tasks were built into a single `retrieval.py` file, each task adding on top of the previous.

### Task 1 — Vector Search
- Connects to ChromaDB using the same embedding model as ingest.py
- Takes a question, converts it to numbers, finds closest chunks
- Returns top 5 results with similarity scores
- Scores: 0.0-1.0, higher = more similar

**Testing issue found**: You tested with completely wrong questions:
- "Which is the 1st element in periodic table?" → Score: 0.24 (very low, correct behavior)
- "What is operation precedence in Java?" → Score: 0.43
- These are NOT FastAPI questions — the low scores proved refusal working correctly
- Fix: Use real FastAPI questions like "How do I create a POST route in FastAPI?"

### Task 2 — BM25 Hybrid Search
- Added `rank_bm25` library for keyword-based search
- BM25 = classic search algorithm used by Elasticsearch
- Tokenizes all chunks once at startup
- Merges vector scores + BM25 scores with 50/50 weighting formula:
  ```
  final_score = (0.5 × vector_score) + (0.5 × bm25_score)
  ```
- Both scores normalized to 0-1 before combining
- Results printed side by side: vector-only vs BM25-only vs hybrid

**Extra library needed**: `pip install numpy`

### Task 3 — Reranker
- Added `CrossEncoder` from `sentence_transformers`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Takes (question, chunk) pairs and scores them together
- Much more accurate than vector search because it reads both simultaneously
- Downloads ~80MB model on first run
- Scores are NOT 0-1 — can be any number (e.g. 8.2, -3.1), higher = better
- After reranking, chunk ORDER changes — better chunks bubble up

### Task 4 — Query Rewriting
- Uses Groq (llama-3.3-70b) to rewrite vague questions before searching
- Example: "make it faster?" → "FastAPI performance optimization async await"
- `temperature=0.1` for consistent, non-creative rewrites
- `max_tokens=100` — rewritten query should be short
- Rewritten query used for SEARCH, original question used for RERANKING and GENERATION
- Introduced `full_pipeline()` function combining all steps

### Task 5 — Answer Generation with Citations
- Passes top reranked chunks + question to Groq
- Chunks numbered [1][2][3] in the prompt
- System prompt instructs: cite every sentence, never make up facts
- Refusal system: if best chunk score < `RELEVANCE_THRESHOLD` (-5.0), return "I don't know"
- `temperature=0.2` for factual, consistent answers
- `max_tokens=1000`

**Full pipeline after Day 2:**
```
User question
    ↓ Groq rewrites it
    ↓ Hybrid search (vector + BM25) on rewritten query
    ↓ Reranker picks truly best 5 from top 10
    ↓ Groq generates cited answer from those 5 chunks
```

---

## 9. DAY 3 — STREAMLIT WEB APP (5 Tasks, all in app.py)

### Task 1 — Basic chat interface
- `st.set_page_config(layout="wide")` for wide layout
- `@st.cache_resource` decorator — loads models ONCE, caches them
- `st.session_state.messages` — stores chat history across page refreshes
- `st.chat_input()` — text box at bottom of screen
- `st.chat_message()` — displays user/assistant bubbles
- `st.spinner()` — shows loading message while processing

### Task 2 — Inline citations
- Answer already has [1][2][3] from retrieval.py
- `st.markdown(answer)` renders the text nicely
- Sources listed below with filenames and scores

### Task 3 — Side panel
- `with st.sidebar:` creates the sidebar
- `st.expander()` makes each chunk collapsible
- Shows filename, reranker score, vector score, BM25 score, chunk preview
- Score colors: 🟢 above 3, 🟡 above 0, 🔴 below 0

### Task 4 — Comparison toggle
- `st.radio()` with options: "Hybrid + Reranker" and "Vector Only"
- `horizontal=True` shows buttons side by side
- Different pipeline runs based on selected mode
- Vector-only mode skips BM25 and reranker

### Task 5 — Refusal styling
- Detects refusal when `used_chunks` is empty
- `st.warning(answer)` shows yellow warning box
- Normal answers use `st.markdown(answer)`

### Issues encountered:

**Issue 1: Sidebar collapsed by default**
- The `>>` arrow in top-left expands it
- Noted as something to fix before reviewer sees it

**Issue 2: App answering "I could not find this" for valid FastAPI questions**
- Symptom: "How do I create a POST route in FastAPI?" → refusal message
- Source scores showing: -1.98, -3.93, -4.72 etc.
- Root cause: `RELEVANCE_THRESHOLD = -5.0` was too strict for this reranker
- Fix: Change `RELEVANCE_THRESHOLD = -5.0` to `RELEVANCE_THRESHOLD = -10.0`
- Also updated `generate_answer` prompt to be more assertive: "You MUST answer using provided chunks. Even if chunks only partially answer, use what is there."

**Issue 3: `ModuleNotFoundError: No module named 'torchvision'`**
- Appeared when running `streamlit run app.py`
- Cause: transformers library trying to load image processing module
- Fix options given in order: `pip install torchvision`, reinstall sentence-transformers, or `pip install transformers==4.40.0`

### Successful app screenshot described:
- Title: "⚡ FastAPI Documentation Assistant"
- Subtitle: "Ask any question about FastAPI. Answers come with citations from the official docs."
- Toggle buttons: "Hybrid + Reranker" (selected) and "Vector Only"
- Info bar: "Full pipeline: Query rewriting → Hybrid search (vector + BM25) → Reranker → Answer"
- Chat bubble showing question: "what is Fast API? where does it benefit a user"
- Answer with [1][2][3] citations inline
- Sources listed: fastapicloud.md, benchmarks.md, oauth2-jwt.md, async.md, release-notes.md

---

## 10. REQUIREMENTS CHECKLIST (assessed mid-project)

**Score: 8/11 required features built**

### Fully built ✅:
- ingest.py (idempotent)
- Hybrid retrieval with documented 50/50 scoring
- Reranker (cross-encoder)
- Query rewriting
- Citations (inline + source list)
- Streamlit UI with chat history
- Comparison toggle
- Refusal for out-of-scope questions

### Partially built ⚠️:
- Side panel (exists but collapsed by default — not visible to reviewer)

### Not built yet ❌:
- 25-question golden set (questions.json)
- RAGAS evaluation scores

---

## 11. DAY 4 — EVALUATION + POLISH + SUBMIT

### Task 1 — questions.json (25 questions)

Full file generated with exactly this structure:
```json
{
  "metadata": {
    "corpus": "FastAPI Documentation",
    "total_questions": 25,
    "categories": {
      "easy": 5, "medium": 8, "hard": 7,
      "multi_concept": 3, "out_of_scope": 2
    }
  },
  "questions": [...]
}
```

**5 easy questions** (IDs 1-5):
1. What is FastAPI?
2. What decorator creates a GET route?
3. How do you install FastAPI?
4. How do you run a FastAPI application?
5. What URL accesses the automatic API docs?

**8 medium questions** (IDs 6-13):
6. How do you declare path parameters?
7. What is a Pydantic model?
8. How do you declare query parameters?
9. How do you handle HTTP exceptions?
10. What are background tasks?
11. How do you add CORS middleware?
12. How do you return a specific HTTP status code?
13. How do you receive form data?

**7 hard questions** (IDs 14-20):
14. Difference between @app.get and APIRouter?
15. How does dependency injection work?
16. What is response_model?
17. How do you handle file uploads?
18. Difference between async def and def route functions?
19. How do you use yield in FastAPI dependencies?
20. How do you add custom middleware?

**3 multi-concept questions** (IDs 21-23):
21. How do you implement OAuth2 with JWT?
22. How do you structure a large FastAPI application?
23. How do you write tests including authenticated endpoints?

**2 out-of-scope questions** (IDs 24-25):
24. "What is the capital of France?" — expected_answer: "NOT_IN_DOCS"
25. "How do I reverse a linked list in Python?" — expected_answer: "NOT_IN_DOCS"

### Task 2 — evaluate.py

Complete script that:
1. Loads all resources (ChromaDB, BM25, reranker, Groq)
2. Reads questions.json
3. Loops through all 25 questions running full pipeline on each
4. Waits 1 second between questions to avoid Groq rate limiting (30 req/min limit)
5. Handles out-of-scope questions separately (checks if system correctly refused)
6. Builds HuggingFace Dataset for RAGAS
7. Runs RAGAS with 3 metrics: faithfulness, answer_relevancy, context_precision
8. Saves results to results.json

**RAGAS metrics explained:**
- **Faithfulness** (0-1): Does the answer only use facts from retrieved chunks? Catches hallucination.
- **Answer Relevancy** (0-1): Does the answer actually address the question?
- **Context Precision** (0-1): Were the right chunks retrieved?

**Good scores:**
```
Faithfulness      : 0.85  (above 0.7 is acceptable)
Answer Relevancy  : 0.79
Context Precision : 0.72
```

**Important note about RAGAS**: RAGAS uses OpenAI by default for evaluation. Workaround: set `OPENAI_API_KEY=your_groq_key_here` in .env.

### Task 3 — Fix worst 3 questions
- Look at results.json for lowest scores
- Common fixes: adjust chunk size, adjust TOP_K, improve generation prompt
- Save before AND after RAGAS scores for README

### Task 4 — README.md
Must contain:
1. What the app does (2 sentences)
2. Requirements (Python version, pip installs)
3. Setup steps (clone corpus, create .env, add keys)
4. How to run (3 commands: ingest → evaluate → streamlit)
5. RAGAS scores before and after
6. Architecture paragraph

### Task 5 — Final GitHub push

**Required files in task1/:**
```
task1/
├── ingest.py
├── retrieval.py
├── app.py
├── evaluate.py
├── questions.json
├── README.md
└── requirements.txt
```

**Must NOT push:** `.env`, `chroma_db/`, `fastapi/`

**Create requirements.txt:**
```
pip freeze > requirements.txt
```

**Final push:**
```
cd D:\AI_Development\ai-intern-assessment-syed
git add .
git commit -m "Task 1 complete - RAG system with RAGAS evaluation"
git push origin main
```

---

## 12. THE GENERIC DOCUMENT UPLOADER QUESTION

You asked how to make the app generic so users can upload any documents. Answer given:

**New libraries needed:** `pip install pypdf python-docx`

**Core idea:**
```python
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(file)
            text = " ".join(page.extract_text() for page in reader.pages)
        else:
            text = file.read().decode("utf-8")
        # then chunk → embed → add to ChromaDB
```

**Recommendation given**: Finish the recruiter's task first. Generic uploader is a bonus feature — build it after you get the job.

---

## 13. WHY THIS IS BETTER THAN CHATGPT/CLAUDE

The one-sentence answer given:
> "ChatGPT generates answers from memory and can be wrong without knowing it — my system retrieves the ground truth first, then generates an answer strictly from that retrieved evidence, so every claim is traceable and verifiable."

**Full comparison table:**

| | ChatGPT/Claude | Your RAG App |
|---|---|---|
| Knowledge source | Trained on internet up to cutoff | Searches exact documentation you ingested |
| Accuracy | Can hallucinate confidently | Answers only from retrieved chunks |
| Citations | Rarely tells source | Every sentence has [1][2][3] with source file |
| Refusal | Often tries to answer anyway | Says "I don't know" when not in docs |
| Up to date | Frozen at training cutoff | Re-run ingest.py to update |
| Trust | Take its word for it | Reviewer can verify against source chunk |
| Domain focus | Knows everything — unfocused | Knows only FastAPI — laser focused |

---

## 14. YOUR TECHNICAL ENVIRONMENT

- **OS**: Windows
- **Python**: 3.14 (path: `C:\Users\nabha\AppData\Local\Programs\Python\Python314\`)
- **Project path**: `D:\AI_Development\ai-intern-assessment-syed\task1\`
- **GitHub username**: SyedNabhan
- **GitHub repo**: `https://github.com/SyedNabhan/ai-intern-assessment-syed.git`
- **Terminal used**: Windows Command Prompt (cmd)
- **Editor**: VS Code

---

## 15. ALL FILES CREATED IN THIS PROJECT

| File | Day | Purpose |
|---|---|---|
| `ingest.py` | Day 1 | Reads FastAPI docs, chunks, embeds, saves to ChromaDB |
| `retrieval.py` | Day 2 | Full pipeline: rewrite → hybrid → rerank → generate |
| `app.py` | Day 3 | Streamlit web app with all 5 UI features |
| `questions.json` | Day 4 | 25 test questions for RAGAS evaluation |
| `evaluate.py` | Day 4 | Runs RAGAS evaluation, saves results.json |
| `README.md` | Day 4 | How to install and run the project |
| `requirements.txt` | Day 4 | All pip packages (`pip freeze > requirements.txt`) |

---

## 16. KEY NUMBERS

- **153** markdown files in FastAPI docs corpus
- **3869** total chunks saved to ChromaDB
- **500** characters per chunk with 50-character overlap
- **TOP_K = 10** — chunks fetched from hybrid search
- **TOP_K_RERANKED = 5** — chunks kept after reranking
- **RELEVANCE_THRESHOLD = -10.0** — final value after bug fix
- **0.5 / 0.5** — vector vs BM25 weighting in hybrid search
- **llama-3.3-70b-versatile** — Groq model used
- **sentence-transformers/all-MiniLM-L6-v2** — embedding model
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranker model
- **25** test questions (5 easy, 8 medium, 7 hard, 3 multi-concept, 2 out-of-scope)
- **localhost:8501** — Streamlit app URL

ENDOFFILE