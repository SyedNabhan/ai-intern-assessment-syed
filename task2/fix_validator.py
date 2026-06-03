with open("agent.py", "r", encoding="utf-8") as f:
    content = f.read()

start = content.index("def input_validator")
end = content.index("# NODE 2 — PLANNER")

new_func = '''def input_validator(state: AgentState) -> dict:
    print(">> [Node 1] input_validator")
    query = state.get("query", "").strip()

    if not query:
        return {"error_message": "Query is empty.", "is_complete": True}

    alpha_chars = [c for c in query if c.isalpha()]
    if len(alpha_chars) == 0:
        print("   REJECTED: no letters")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    vowels = sum(c in "aeiouAEIOU" for c in alpha_chars)
    vowel_ratio = vowels / len(alpha_chars)
    if vowel_ratio < 0.15:
        print("   REJECTED: gibberish")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    words = query.split()
    if all(len(set(w.lower())) <= 1 for w in words if w.isalpha()):
        print("   REJECTED: repeated letters")
        return {"error_message": "Query appears invalid. Please enter a clear research question.", "is_complete": True}

    if len(words) == 1:
        print("   REJECTED: too vague")
        return {
            "needs_clarification": True,
            "clarification_question": f"'{query}' is too vague. What specifically about '{query}' do you want to research?",
            "is_complete": True,
        }

    if len(query) < 5:
        return {"error_message": "Query is too short.", "is_complete": True}

    print(f"   Query accepted: {query[:60]}")
    return {}


'''

content = content[:start] + new_func + content[end:]

with open("agent.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done! agent.py patched successfully.")