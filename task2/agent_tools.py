# tools.py
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS


# ── TOOL 1 ─────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches DuckDuckGo for the given query.
    No API key needed.

    Input:
      query (str): the search query
      max_results (int): how many results to return (default 5)

    Output:
      list of dicts with keys: title, url, snippet

    Example call: web_search("latest LLM research 2025")
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        return [{"error": str(e), "title": "", "url": "", "snippet": ""}]


# ── TOOL 2 ─────────────────────────────────────────────────────
def scrape_page(url: str, max_chars: int = 3000) -> dict:
    """
    Fetches a web page and returns its cleaned text content.
    Handles timeouts and errors gracefully.

    Input:
      url (str): full URL of the page to scrape
      max_chars (int): truncate content to this length (default 3000)

    Output:
      dict with keys: url, content, error (error is None if successful)

    Example call: scrape_page("https://en.wikipedia.org/wiki/LLM")
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Clean up blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        content = "\n".join(lines)[:max_chars]

        return {"url": url, "content": content, "error": None}

    except Exception as e:
        return {"url": url, "content": "", "error": str(e)}


# ── TOOL 3 ─────────────────────────────────────────────────────
def query_rag(question: str) -> dict:
    try:
        import sys
        import os

        task1_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'task1')
        )
        if task1_path not in sys.path:
            sys.path.insert(0, task1_path)

        # ── KEY FIX: change working directory to task-1 ──
        original_dir = os.getcwd()
        os.chdir(task1_path)

        try:
            from retrieval import (
                load_collection,
                build_bm25_index,
                load_reranker,
                load_groq_client,
                full_pipeline,
            )

            collection, all_chunks, all_metadatas, all_ids = load_collection()
            bm25_index  = build_bm25_index(all_chunks)
            reranker    = load_reranker()
            groq_client = load_groq_client()

            answer, used_chunks, rewritten_query = full_pipeline(
                groq_client,
                collection,
                bm25_index,
                reranker,
                all_chunks,
                all_metadatas,
                all_ids,
                question,
            )

        finally:
            # Always restore original directory
            os.chdir(original_dir)

        return {
            "question": question,
            "answer": answer,
            "chunks": used_chunks,
            "rewritten_query": rewritten_query,
            "error": None,
        }

    except Exception as e:
        return {
            "question": question,
            "answer": "",
            "chunks": [],
            "rewritten_query": question,
            "error": f"RAG system unavailable: {str(e)}",
        }
    
# ── TOOL 4 ─────────────────────────────────────────────────────
def format_report(
    query: str,
    search_results: list[dict],
    scraped_pages: list[dict],
    rag_results: list[dict],
    tool_errors: list[str],
) -> str:
    """
    Takes all accumulated results and formats them into a
    clean markdown report. Pure function — no LLM call.

    Input:
      query: original user question
      search_results: from web_search
      scraped_pages: from scrape_page
      rag_results: from query_rag
      tool_errors: any errors that occurred

    Output:
      formatted markdown string

    Example call: format_report(query="...", search_results=[...], ...)
    """
    sections = []
    sections.append(f"# Research Report\n**Query:** {query}\n")

    if search_results:
        sections.append("## Web Search Results")
        for r in search_results:
            if r.get("error"):
                continue
            sections.append(f"- **{r['title']}**\n  {r['snippet']}\n  Source: {r['url']}")

    if scraped_pages:
        sections.append("\n## Scraped Pages")
        for p in scraped_pages:
            if p.get("error"):
                sections.append(f"- {p['url']} — failed: {p['error']}")
            else:
                sections.append(f"- **{p['url']}**\n```\n{p['content'][:500]}\n```")

    if rag_results and rag_results.get("answer"):
        sections.append("\n## Documentation Results (Multi Docs)")
        sections.append(rag_results["answer"])
        if rag_results.get("chunks"):
            sections.append("**Sources:**")
            for chunk in rag_results["chunks"]:
                sections.append(f"- {chunk.get('source', 'unknown')} (score: {chunk.get('reranker_score', 'N/A')})")

    if tool_errors:
        sections.append("\n## Errors Encountered")
        for e in tool_errors:
            sections.append(f"- {e}")

    return "\n\n".join(sections)


# ── QUICK TEST ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing web_search...")
    results = web_search("large language models 2025", max_results=3)
    for r in results:
        print(f"  - {r['title']}: {r['url']}")

    print("\nTesting scrape_page...")
    page = scrape_page("https://en.wikipedia.org/wiki/Large_language_model")
    print(f"  Content length: {len(page['content'])} chars")
    print(f"  Error: {page['error']}")

    print("\nTesting query_rag...")
    rag = query_rag("how does indexing work?")
    print(f"  RAG error (expected if task-1 not connected): {rag['error']}")

    print("\nTesting format_report...")
    report = format_report(
        query="test query",
        search_results=results,
        scraped_pages=[page],
        rag_results={"results": []},
        tool_errors=[],
    )
    print(report[:300])