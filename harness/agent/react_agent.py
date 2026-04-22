from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.agent.prompts import build_system_prompt
from harness.providers.http import post_json, strip_think_tags

log = logging.getLogger(__name__)


@dataclass
class AgentStep:
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    query: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


def parse_react_output(text: str) -> dict[str, Any]:
    """Parse Thought / Action / Action Input / Final Answer from LLM output.

    :param text: Raw LLM response.
    :returns: Dict with keys ``thought``, ``action``, ``action_input``,
        ``final_answer``.
    """
    result: dict[str, Any] = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", text, re.DOTALL
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    action_match = re.search(r"Action:\s*(\S+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
    if input_match:
        result["action_input"] = _extract_json(input_match.group(1))

    return result


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from *text*, handling nested braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


class _BaseReactAgent(ABC):
    """Base ReAct loop. Subclasses provide :meth:`_chat`."""

    def __init__(self, mem: Any, max_steps: int = 5, system_prompt: str | None = None):
        self._mem = mem
        self._max_steps = max_steps
        self._system_prompt = system_prompt or build_system_prompt(
            mem.get_tool_definitions()
        )

    def run(self, query: str) -> AgentResult:
        """Run the ReAct loop for a user query.

        :param query: Natural language question.
        :returns: Agent result with answer, steps, and tools used.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        result = AgentResult(query=query, answer="")

        for _ in range(self._max_steps):
            response = self._chat(messages)
            parsed = parse_react_output(response)
            step = AgentStep(thought=parsed["thought"])

            if parsed["final_answer"]:
                result.answer = parsed["final_answer"]
                result.steps.append(step)
                break

            if parsed["action"]:
                step.action = parsed["action"]
                step.action_input = parsed["action_input"] or {}
                result.tools_used.append(parsed["action"])

                try:
                    observation = self._mem.dispatch_tool_call(
                        parsed["action"], step.action_input
                    )
                except Exception as e:
                    observation = f"Error: {e}"

                step.observation = observation
                result.steps.append(step)
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })
            else:
                result.answer = response
                result.steps.append(step)
                break
        else:
            last = result.steps[-1].thought if result.steps else "none"
            result.answer = (
                f"Reached max steps ({self._max_steps}). Last thought: {last}"
            )

        return result

    @abstractmethod
    def _chat(self, messages: list[dict[str, str]]) -> str: ...


class ReactAgent(_BaseReactAgent):
    """ReAct agent backed by Ollama.

    :param mem: :class:`~emem.SpatioTemporalMemory` instance.
    :param model: Ollama model name.
    :param base_url: Ollama server URL.
    :param max_steps: Maximum ReAct iterations.
    :param system_prompt: Custom system prompt. If ``None``, uses the default
        ReAct prompt built from the memory's tool definitions.
    """

    def __init__(
        self,
        mem: Any,
        model: str = "qwen3.5:latest",
        base_url: str = "http://localhost:11434",
        max_steps: int = 5,
        system_prompt: str | None = None,
        think: bool = False,
        seed: int | None = None,
    ):
        super().__init__(mem, max_steps, system_prompt)
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._think = think
        self._seed = seed

    def _chat(self, messages: list[dict[str, str]]) -> str:
        from harness.providers.http import post_json
        from harness.providers.ollama_vlm import _is_thinking_model

        body: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if self._seed is not None:
            body["options"] = {"seed": self._seed}
        if _is_thinking_model(self._model):
            body["think"] = self._think
        data = post_json(self._url, body, timeout=500)
        return strip_think_tags(data["message"]["content"])


class GeminiReactAgent(_BaseReactAgent):
    """ReAct agent backed by Google Gemini.

    :param mem: :class:`~emem.SpatioTemporalMemory` instance.
    :param model: Gemini model name.
    :param api_key: Gemini API key (falls back to ``GEMINI_API_KEY`` env var).
    :param max_steps: Maximum ReAct iterations.
    :param system_prompt: Custom system prompt. If ``None``, uses the default
        ReAct prompt built from the memory's tool definitions.
    """

    def __init__(
        self,
        mem: Any,
        model: str = "gemini-2.0-flash-lite",
        api_key: str | None = None,
        max_steps: int = 5,
        system_prompt: str | None = None,
    ):
        super().__init__(mem, max_steps, system_prompt)
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key required: pass api_key= or set GEMINI_API_KEY"
            )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        from harness.providers.http import post_json_with_retry

        # Convert OpenAI-style messages to Gemini contents format
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
            else:
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

        url = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{self._model}:generateContent?key={self._api_key}"
        )
        data = post_json_with_retry(url, {"contents": contents})
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""
        return strip_think_tags(text)


_DEFAULT_NATIVE_SYSTEM_PROMPT = (
    "You are a memory-augmented robot assistant. You have access to a "
    "spatio-temporal memory system via tool calls. Use the provided tools "
    "to look up relevant information before answering. Prefer chaining "
    "tools when a single call does not fully answer the question. When "
    "you have enough information, respond with a concise final answer "
    "and do not emit any further tool calls. Do not invent facts that "
    "were not returned by a tool."
)


def ollama_model_supports_tools(model: str, base_url: str) -> bool:
    """Probe ``/api/show`` to check whether an Ollama model declares the
    ``tools`` capability.

    :param model: Model tag, e.g. ``"qwen3.5:27b"``.
    :param base_url: Ollama server URL.
    :returns: ``True`` if the model's capabilities include ``"tools"``,
        ``False`` otherwise (including when the probe fails).
    """
    try:
        data = post_json(
            f"{base_url.rstrip('/')}/api/show",
            {"name": model},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ollama /api/show probe failed for %s: %s", model, exc)
        return False
    caps = data.get("capabilities") or []
    return "tools" in caps


def _coerce_tool_args(args: Any) -> dict[str, Any]:
    """Normalise the ``arguments`` field Ollama returns inside a tool-call.

    Ollama usually returns a parsed dict, but occasionally older builds
    pass the arguments back as a JSON-encoded string.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class NativeToolCallAgent:
    """ReAct-style agent backed by Ollama's native ``tools`` API.

    Unlike :class:`ReactAgent`, this agent does not rely on parsing
    ``Thought: / Action:`` text. Instead it sends the tool schema via
    the Ollama ``tools`` field, reads back ``message.tool_calls``
    structurally, and iterates until the model stops calling tools.
    This is the recommended path for models like Qwen 3.5 and Gemma 4
    that expose ``"tools"`` in their capabilities.

    :param mem: Memory instance (or :class:`_AblatedMemory` wrapper).
    :param model: Ollama model tag.
    :param base_url: Ollama server URL.
    :param max_steps: Maximum tool-call rounds before giving up.
    :param system_prompt: Custom system prompt. A short default is used
        when ``None``.
    :param seed: Optional sampler seed (Ollama ``options.seed``).
    """

    def __init__(
        self,
        mem: Any,
        model: str = "qwen3.5:latest",
        base_url: str = "http://localhost:11434",
        max_steps: int = 5,
        system_prompt: str | None = None,
        seed: int | None = None,
        **_: Any,  # swallow ReAct-only kwargs (e.g. ``think``) for symmetry
    ):
        self._mem = mem
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._max_steps = max_steps
        self._seed = seed
        self._system_prompt = system_prompt or _DEFAULT_NATIVE_SYSTEM_PROMPT
        self._tools = mem.get_tool_definitions()

    def run(self, query: str) -> AgentResult:
        """Run the tool-calling loop for a user query."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        result = AgentResult(query=query, answer="")

        for _ in range(self._max_steps):
            raw = self._chat(messages)
            msg = raw.get("message", {}) or {}
            content = strip_think_tags(msg.get("content") or "")
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                result.answer = content
                result.steps.append(AgentStep(thought=content))
                break

            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_entry)

            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or ""
                args = _coerce_tool_args(fn.get("arguments"))

                if not name:
                    continue

                result.tools_used.append(name)
                try:
                    observation = self._mem.dispatch_tool_call(name, args)
                except Exception as exc:  # noqa: BLE001
                    observation = f"Error: {exc}"

                result.steps.append(
                    AgentStep(
                        thought=content,
                        action=name,
                        action_input=args,
                        observation=observation,
                    )
                )
                messages.append({
                    "role": "tool",
                    "content": str(observation),
                    "name": name,
                })
        else:
            last = result.steps[-1].thought if result.steps else "none"
            result.answer = (
                f"Reached max steps ({self._max_steps}). Last thought: {last}"
            )

        return result

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """POST to Ollama ``/api/chat`` with the ``tools`` field populated."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": self._tools,
            "stream": False,
        }
        if self._seed is not None:
            body["options"] = {"seed": self._seed}
        return post_json(self._url, body, timeout=500)
