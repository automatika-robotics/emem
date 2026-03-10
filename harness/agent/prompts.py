from typing import Any

_SYSTEM_TEMPLATE = """\
You are a memory-augmented robot assistant. You have access to a spatio-temporal \
memory system with the following tools:

{tools}

To use a tool, respond in EXACTLY this format:

Thought: <your reasoning about what to do>
Action: <tool_name>
Action Input: <JSON object with tool parameters>

After you receive the tool's output as an Observation, you may continue with \
another Thought/Action cycle or provide a final answer.

When you have enough information to answer the user's question, respond with:

Thought: I now have enough information to answer.
Final Answer: <your answer>

Rules:
- Always start with a Thought before taking an Action.
- Action Input must be valid JSON.
- Use the most specific tool for the query.
- You may chain multiple tool calls if needed.
- Be concise in your Final Answer.
"""


def build_system_prompt(tool_definitions: list[dict[str, Any]]) -> str:
    """Build the ReAct system prompt from eMEM tool definitions.

    :param tool_definitions: Output of
        :meth:`~emem.SpatioTemporalMemory.get_tool_definitions`.
    :returns: Complete system prompt string.
    """
    parts: list[str] = []
    for tool in tool_definitions:
        name = tool["name"]
        desc = tool["description"]
        params = tool.get("parameters", {}).get("properties", {})
        required = set(tool.get("parameters", {}).get("required", []))

        param_lines: list[str] = []
        for pname, pschema in params.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req = " (required)" if pname in required else ""
            param_lines.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        param_block = "\n".join(param_lines) if param_lines else "    (no parameters)"
        parts.append(f"  {name}: {desc}\n  Parameters:\n{param_block}")

    return _SYSTEM_TEMPLATE.format(tools="\n\n".join(parts))
