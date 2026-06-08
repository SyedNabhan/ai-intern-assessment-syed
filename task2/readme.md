# Task 2 — Tool-Using Research Agent

A LangGraph-powered autonomous research agent that takes a user's question, searches the web, scrapes pages, queries the Task 1 RAG system, and produces a structured markdown report with citations. Built with a live Streamlit interface that streams the agent's reasoning step by step.

---

## What It Does

The user types a research question. The agent:
1. Validates the input
2. Plans which tool to use
3. Searches the web (DuckDuckGo)
4. Optionally scrapes pages or queries the Task 1 RAG system
5. Synthesizes all findings into a structured report using Llama 3.3

All steps are visible in real time in the UI and recorded in Langfuse.

---

## Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq API — `llama-3.3-70b-versatile` (free) |
| Web Search | `ddgs` — DuckDuckGo, no API key needed |
| Web Scraping | `httpx` + `BeautifulSoup4` |
| RAG Integration | Task 1 retrieval pipeline (ChromaDB + BM25) |
| Observability | Langfuse v4.7.1 |
| UI | Streamlit |
| Dependency Management | `uv` |

---

## Project Structure

```
task-2/
├── agent_state.py       # TypedDict state shared across all nodes
├── agent_tools.py       # 4 tools: web_search, scrape_page, query_rag, format_report
├── agent.py             # LangGraph graph, nodes, conditional edge, runner
├── app.py               # Streamlit UI with live streaming
├── decisions.md         # Design decisions and tradeoffs
├── guardrail_tests/
│   └── test_guardrails.py   # 3 guardrail tests
├── runs/
│   ├── run_1_happy_path.json
│   ├── run_2_invalid_input.json
│   └── run_3_vague_input.json
├── task1/               # Task 1 RAG pipeline (required for query_rag)
├── chroma_db/           # ChromaDB vector store from Task 1
├── .env                 # API keys (never committed)
└── pyproject.toml       # uv dependencies
```

---

## Setup

### 1. Clone and navigate

```cmd
cd task-2
```

### 2. Install dependencies

```cmd
uv sync
```

### 3. Set up environment variables

Create a `.env` file in `task-2/`:

```
GROQ_API_KEY=your_groq_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Get your keys from:
- Groq: `console.groq.com` (free)
- Langfuse: `cloud.langfuse.com` (free)

### 4. Launch the UI

```cmd
uv run streamlit run app.py
```

---

## Agent Graph

```
[input_validator]
       ↓
   [planner]
       ↓
 [tool_caller] ←─────────────┐
       ↓                     │
 [should_continue] ──loop────┘
       ↓ (done / limit hit)
  [synthesizer]
       ↓
     [END]
```

### Nodes

| Node | What it does |
|---|---|
| `input_validator` | Rejects empty, gibberish, or single-word queries |
| `planner` | Decides which tool to call next based on current state |
| `tool_caller` | Executes the chosen tool, handles failures gracefully |
| `synthesizer` | Writes the final structured report using Llama 3.3 |

### Conditional Edge

`should_continue` routes back to `tool_caller` or forward to `synthesizer` based on:
- Iteration count ≥ 3
- Search results collected ≥ 3
- Cost limit of $1.00 reached
- Planner marks task complete

---

## Tools

### `web_search(query, max_results=5)`
Searches DuckDuckGo. No API key required. Returns list of `{title, url, snippet}`.

### `scrape_page(url, max_chars=3000)`
Fetches and cleans a web page using httpx + BeautifulSoup. Returns `{url, content, error}`. Handles timeouts and 4xx/5xx errors gracefully.

### `query_rag(question)`
Queries the Task 1 RAG pipeline (FastAPI documentation corpus). Uses ChromaDB + BM25 hybrid retrieval + reranker. Falls back gracefully if Task 1 is unavailable.

### `format_report(query, search_results, scraped_pages, rag_results, tool_errors)`
Pure function — no LLM call. Formats all accumulated results into a structured markdown document passed to the synthesizer.

---

## Guardrails

| Failure Mode | Input Example | Behaviour |
|---|---|---|
| Invalid input | `???###!!!` | Returns error message, skips to END |
| Vague input | `AI` | Returns clarification question, skips to END |
| Tool failure | Network timeout | Logs error in state, agent continues with partial results |
| Cost ceiling | Any run > $1.00 | Agent stops immediately, synthesizes with data so far |
| Iteration limit | Any run > 3 loops | Agent stops, synthesizes with data so far |

---

## Running Tests

```cmd
uv run pytest guardrail_tests/
```

---

## Example Runs

All saved in `runs/`:

**Happy path** — `run_1_happy_path.json`
> "What are the latest developments in large language models?"
→ Full report with key findings and cited sources

**Invalid input** — `run_2_invalid_input.json`
> `???###!!!`
→ `"Query appears invalid. Please enter a clear research question."`

**Vague input** — `run_3_vague_input.json`
> `AI`
→ `"'AI' is too vague. What specifically about 'AI' do you want to research?"`

---

## Observability

Every run is traced in Langfuse with:
- Each node entered and exited
- Tool called with arguments and result summary
- Iteration count at each step
- Final report in output

View traces at `cloud.langfuse.com` after running.
