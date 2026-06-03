"""
Task 2 — Guardrail Tests
Run with: uv run pytest guardrail_tests/ -v
"""

import pytest
import sys
import os

# Make sure the task-2 root is on the path so agent.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import build_graph  # type: ignore  # noqa: E402
graph = build_graph()


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_initial_state(query: str) -> dict:
    """Return a fresh initial state for a given query."""
    return {
        "query": query,
        "search_results": [],
        "scraped_content": [],
        "rag_results": [],
        "tool_errors": [],
        "iterations": 0,
        "total_tokens": 0,
        "final_report": None,
        "error_message": None,
        "needs_clarification": False,
        "clarification_question": None,
        "cost_usd": 0.0,
    }


def run_to_completion(query: str) -> dict:
    """Run the graph to completion and return the final accumulated state."""
    state = make_initial_state(query)
    final = dict(state)

    for event in graph.stream(state, stream_mode="updates"):
        for _node, updates in event.items():
            if updates is not None:
                final.update(updates)

    return final


# ── Test 1: Invalid / gibberish input ─────────────────────────────────────────

class TestInvalidInput:
    """
    Guardrail: the input_validator node should reject clearly invalid queries
    (empty strings, random gibberish, pure symbols) and set error_message.
    The agent must NOT proceed to the planner.
    """

    @pytest.mark.parametrize("bad_query", [
        "asdfghjkl",
        "!@#$%^&*()",
        "zzzzzzzzzzzzz",
        "123 456 789",
        "aaa bbb ccc ddd eee fff",   # random tokens, no real meaning
    ])
    def test_gibberish_sets_error_message(self, bad_query):
        final = run_to_completion(bad_query)
        assert final["error_message"] is not None, (
            f"Expected error_message to be set for gibberish query: {bad_query!r}"
        )

    def test_gibberish_does_not_produce_report(self):
        final = run_to_completion("asdfghjkl zxcvbnm qwrtyp")
        assert final["error_message"] is not None, (
            "Agent should set error_message for an invalid query."
        )

    def test_empty_query_sets_error(self):
        final = run_to_completion("   ")
        assert final["error_message"] is not None, (
            "A blank/whitespace-only query should set error_message."
        )


# ── Test 2: Vague / single-word input needing clarification ───────────────────

class TestNeedsClarification:
    """
    Guardrail: a single-word or overly vague query should set needs_clarification=True
    and provide a clarification_question. The agent must not guess or proceed to search.
    """

    @pytest.mark.parametrize("vague_query", [
        "AI",
        "Python",
        "research",
        "help",
        "thing",
    ])
    def test_single_word_triggers_clarification(self, vague_query):
        final = run_to_completion(vague_query)
        assert final["needs_clarification"] is True, (
            f"Expected needs_clarification=True for single-word query: {vague_query!r}"
        )

    def test_clarification_question_is_non_empty(self):
        final = run_to_completion("stuff")
        assert final.get("clarification_question"), (
            "clarification_question must be a non-empty string when needs_clarification is True."
        )

    def test_vague_query_does_not_produce_report(self):
        final = run_to_completion("things")
        assert final["final_report"] is None, (
            "Agent should not produce a report when clarification is needed."
        )

    def test_vague_query_no_error_message(self):
        """Clarification is distinct from an error — error_message should be None."""
        final = run_to_completion("data")
        assert final["error_message"] is None, (
            "A vague query should set needs_clarification, not error_message."
        )


# ── Test 3: Tool failure — agent continues and returns partial result ──────────
class TestToolFailure:
    def test_tool_errors_populated_on_scrape_failure(self):
        """Call tool_caller directly with a broken scrape URL."""
        from agent import tool_caller
        state = make_initial_state("What are transformers in deep learning?")
        state["current_tool"] = "scrape_page"
        state["tool_args"] = {"url": "http://localhost:99999/invalid"}  # guaranteed to fail
        
        result = tool_caller(state)
        assert len(result.get("tool_errors", [])) > 0, (
            "tool_errors should contain at least one entry after a tool failure."
        )

    def test_agent_produces_report_despite_tool_failure(self):
        """Call tool_caller with broken URL, then synthesizer — report still produced."""
        from agent import tool_caller, synthesizer
        state = make_initial_state("Explain the difference between RAG and fine-tuning.")
        state["current_tool"] = "scrape_page"
        state["tool_args"] = {"url": "http://localhost:99999/invalid"}
        
        result = tool_caller(state)
        state.update(result)
        # Add some fake search results so synthesizer has something to work with
        state["search_results"] = [{"title": "RAG vs Fine-tuning", "url": "http://example.com", "snippet": "RAG retrieves context at inference time while fine-tuning bakes knowledge into weights."}]
        
        synth_result = synthesizer(state)
        assert synth_result.get("final_report") is not None, (
            "Synthesizer should produce a report even when a tool failed."
        )

    def test_tool_errors_do_not_crash_graph(self):
        """Broken scrape URL should not raise unhandled exception."""
        from agent import tool_caller
        state = make_initial_state("Compare RLHF and DPO")
        state["current_tool"] = "scrape_page"
        state["tool_args"] = {"url": "http://localhost:99999/invalid"}
        try:
            tool_caller(state)
        except Exception as exc:
            pytest.fail(f"tool_caller raised an unexpected exception: {exc}")
        assert True
