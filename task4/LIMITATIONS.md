# Limitations

1. **Sequential subsystem calls (high effort to fix)**
   RAG, agent, and GitHub run one after another. Parallel async calls
   would cut latency by ~60%. Requires refactoring orchestrator to use
   asyncio.gather() and async-compatible LangGraph invocation.

2. **No conversation memory (medium effort)**
   Each question is independent. A follow-up like "tell me more about
   that" loses the previous context. Needs a message history passed
   into the synthesis prompt and stored in a database between sessions.

3. **Cost tracking is approximate (low effort)**
   Token counts are estimated, not measured. Real tracking requires
   reading response.usage from every Groq API call and maintaining a
   running total across all three subsystems.