from github_client import create_issue

repo = "SyedNabhan/ai-intern-assessment-syed"

create_issue(repo, "Task 1 complete - RAG pipeline", "Documentation Q&A system built with ChromaDB, hybrid retrieval, reranker, and Streamlit UI.", ["task-1"])
create_issue(repo, "Task 2 complete - Research Agent", "LangGraph agent with web search, scraping, RAG tool, Langfuse observability, and Streamlit UI.", ["task-2"])
create_issue(repo, "Bug: ChromaDB version mismatch", "ChromaDB threw TypeError on collection.count() due to version incompatibility. Fixed by upgrading.", ["bug"])
create_issue(repo, "Task 3 in progress - MCP Server", "Building GitHub Issues MCP server with FastMCP and stdio transport.", ["task-3"])
create_issue(repo, "Improvement: async tool calls in agent", "Planner currently calls one tool per iteration. Parallel async calls would reduce latency by 2x.", ["enhancement"])