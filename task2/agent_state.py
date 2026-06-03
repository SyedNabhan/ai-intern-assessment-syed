# agent_state.py
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    query: str
    plan: list
    current_tool: Optional[str]
    tool_args: dict
    search_results: list
    scraped_pages: list
    rag_results: list
    tool_errors: list
    iteration_count: int
    max_iterations: int
    total_cost_usd: float
    cost_limit_usd: float
    is_complete: bool
    needs_clarification: bool
    clarification_question: Optional[str]
    final_report: Optional[str]
    error_message: Optional[str]


def default_state(query: str) -> AgentState:
    return {
        "query": query,
        "plan": [],
        "current_tool": None,
        "tool_args": {},
        "search_results": [],
        "scraped_pages": [],
        "rag_results": [],
        "tool_errors": [],
        "iteration_count": 0,
        "max_iterations": 5,
        "total_cost_usd": 0.0,
        "cost_limit_usd": 1.0,
        "is_complete": False,
        "needs_clarification": False,
        "clarification_question": None,
        "final_report": None,
        "error_message": None,
    }