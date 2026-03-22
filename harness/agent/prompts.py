from typing import Any, Optional

_PREAMBLE = """\
You are a memory-augmented robot assistant. You have access to a spatio-temporal \
memory system with the following tools:"""

_REACT_FORMAT = """\
To use a tool, respond in EXACTLY this format:

Thought: <your reasoning about what to do>
Action: <tool_name>
Action Input: <JSON object with tool parameters>

After you receive the tool's output as an Observation, you may continue with \
another Thought/Action cycle or provide a final answer.

When you have enough information to answer the user's question, respond with:

Thought: I now have enough information to answer.
Final Answer: <your answer>

Tool selection guide — pick ONE tool that best matches the query:
- "What did I see recently?" / "What happened in the last N minutes?" → temporal_query
- "What is near (x, y)?" / "What's around position ...?" → spatial_query
- "Where is X?" / "Find the location of X" → locate
- "Tell me everything about X" / "What do I know about X?" → recall
- "What's my battery/temperature/joint status?" → body_status
- "Summarize my exploration" / "What episodes have there been?" → episode_summary
- "What's the current situation?" / "What's around me right now?" → get_current_context
- "Search for X" / "What places have I visited?" / general search → semantic_search
- "What summaries exist about X?" → search_gists
- "Find entities named X" / "What objects have I seen?" → entity_query

Rules:
- Always start with a Thought before taking an Action.
- Action Input must be valid JSON.
- Prefer answering with a SINGLE tool call. Only chain tools if the first result is genuinely insufficient.
- After receiving an Observation, give a Final Answer unless the result was empty or wrong.
- Be concise in your Final Answer."""


def format_tool_definitions(tool_definitions: list[dict[str, Any]]) -> str:
    """Format eMEM tool definitions as a human-readable block.

    :param tool_definitions: Output of
        :meth:`~emem.SpatioTemporalMemory.get_tool_definitions`.
    :returns: Formatted tool description string.
    """
    parts: list[str] = []
    for tool in tool_definitions:
        fn = tool.get("function", tool)
        name = fn["name"]
        desc = fn["description"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))

        param_lines: list[str] = []
        for pname, pschema in params.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req = " (required)" if pname in required else ""
            param_lines.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        param_block = "\n".join(param_lines) if param_lines else "    (no parameters)"
        parts.append(f"  {name}: {desc}\n  Parameters:\n{param_block}")

    return "\n\n".join(parts)


def build_system_prompt(
    tool_definitions: list[dict[str, Any]],
    preamble: Optional[str] = None,
) -> str:
    """Build the ReAct system prompt from eMEM tool definitions.

    :param tool_definitions: Output of
        :meth:`~emem.SpatioTemporalMemory.get_tool_definitions`.
    :param preamble: Custom preamble text placed before the tool list.
        If ``None``, uses the default robot assistant preamble.
    :returns: Complete system prompt string.
    """
    intro = preamble if preamble is not None else _PREAMBLE
    tools = format_tool_definitions(tool_definitions)
    return f"{intro}\n\n{tools}\n\n{_REACT_FORMAT}"
