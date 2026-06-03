# Task 2 — Design Decisions

## 1. Use Case Choice: Research Assistant over PR Reviewer

I chose the **Research Assistant** because it exercises a broader range of tool types — web search, page scraping, and RAG retrieval — and requires the agent to synthesise information across heterogeneous sources rather than apply a fixed analysis pattern to a single artifact.

The PR Reviewer use case is more deterministic: fetch the diff, lint it, classify findings. That makes guardrail design easier but reduces the agent's need to reason about *which tools to call and in what order*, which is the core skill the assessment is testing. The Research Assistant has genuine branching decisions at each planner step and makes the conditional edge logic non-trivial.

There is also a practical advantage: the Research Assistant naturally reuses the Task 1 RAG pipeline as one of its tools, satisfying the "Task 2 reuses Task 1" requirement in the day-by-day plan without any artificial coupling.

---

## 2. Tool Schema Decisions

### `web_search`
```
web_search(query: str) -> list[dict]
```
The query parameter is typed `str` rather than accepting a structured object because DuckDuckGo's interface is a plain text query. The description emphasises *when* to use this tool — "as the first step for any topic that requires current information or broad coverage" — to distinguish it from `query_rag`, which the planner should prefer for questions about the ingested documentation corpus.

Result format: a list of `{title, url, snippet}` dicts. The planner can inspect titles and snippets to decide which URLs to scrape next, making it a natural first-stage tool.

### `scrape_page`
```
scrape_page(url: str) -> str
```
Returns cleaned plain text (BeautifulSoup strips HTML). The description warns: "Use only on URLs returned by web_search — do not construct speculative URLs." This prevents the planner from hallucinating URLs and then getting 404s.

Error handling is explicit in the description: "Returns an empty string and logs to tool_errors if the request fails or times out." The agent is therefore never surprised by an exception; it sees an empty result and a logged error.

### `query_rag`
```
query_rag(question: str) -> list[dict]
```
The description specifies: "Use this instead of web_search when the question is about [corpus name]. Returns up to 5 ranked chunks with source file and section metadata." The narrow scope prevents the planner from routing general web questions through the RAG system and getting low-quality corpus-only answers.

### `format_report`
```
format_report(results: dict) -> str
```
This is a pure formatting function, not a data-fetching tool. Its description is explicit: "Call this only as the final step, once all search and scrape results are collected. Do not call it mid-run to preview partial results." Without this guardrail in the description, early experiments showed the planner calling it after the first search result and short-circuiting the rest of the research.

---

## 3. State Design: TypedDict over Pydantic BaseModel

The original skeleton used a Pydantic `BaseModel`. This caused a runtime error (`AgentState has no attribute 'get'`) because LangGraph's internal reducer calls `.get()` on state objects, which Pydantic models do not support by default.

Switching to `TypedDict` resolved this immediately. TypedDict is also more idiomatic for LangGraph — the official documentation and most community examples use it — and avoids Pydantic's validation overhead on every state update, which matters when the graph has many short-lived intermediate states.

The only thing lost by dropping Pydantic is runtime field validation. This is acceptable here because the tools are all internal and the types are simple (strings, lists of dicts, booleans). If the agent were exposed as an API endpoint receiving untrusted input, re-introducing Pydantic at the ingress point would make sense.

---

## 4. LLM Choice: Groq + llama-3.3-70b-versatile

**Why Groq:** The Groq inference API is free at the usage levels required for this assessment and significantly faster than the Anthropic or OpenAI APIs on equivalent tasks (sub-second TTFT for the synthesis step). Speed matters for the UI: a slow synthesizer makes the live stream panel feel unresponsive.

**Why llama-3.3-70b-versatile:** The originally planned model (`llama3-70b-8192`) was decommissioned mid-assessment. `llama-3.3-70b-versatile` is its direct replacement with a larger context window (128k tokens), which is important for the synthesis step where all scraped content and search results are passed in a single prompt.

The 70B size is the minimum I would use for synthesis. Smaller models (8B, 13B) produced reports with hallucinated citations and poor source attribution in early experiments — exactly the failure mode that matters most for this use case.

**Why not Claude or GPT-4o:** Cost. At Groq's pricing, a full research run costs under $0.01. The $1 cost ceiling is therefore a stress-test guard rather than a routine concern. Using Claude or GPT-4o would require more careful token counting and would push against the ceiling on long scrapes.

---

## 5. What I Would Improve with More Time

**Async tool calls in the planner.** Currently the planner calls one tool per iteration, which means three iterations for three sources. If the planner could issue all tool calls in parallel (via `asyncio.gather`), latency for a three-source run would drop by roughly 2×. LangGraph supports async nodes; this is the highest-ROI change.

**Smarter query planning.** The current planner uses a simple rule: "call web_search first, then scrape the top result, then query RAG, then stop." A better planner would inspect the question, decide whether RAG is even relevant, and allocate the iteration budget based on question complexity. This would require a small planning LLM call upfront.

**Caching RAG loads.** The `query_rag` tool re-initialises the Chroma client and embedding model on every call via `os.chdir`. A module-level singleton would eliminate this overhead and reduce the per-call latency from ~2 seconds to ~200ms.

**Memory layer.** The agent currently has no memory between runs. A simple key-value store (Redis or SQLite) keyed by a hash of the query could cache reports for repeated questions, avoiding redundant searches and scrapes.

**Better citation extraction.** The synthesiser currently asks the LLM to generate citations from the content it received. A more reliable approach would be to inject source IDs into the content before synthesis and then parse them back out of the response, reducing hallucinated or misattributed citations.
