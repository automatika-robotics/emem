"""Tests for ReAct agent — parsing tests (no deps) + integration (@pytest.mark.ollama)."""

import pytest

from harness.agent.react_agent import parse_react_output


class TestParseReactOutput:
    def test_thought_and_action(self):
        text = (
            "Thought: I need to search memory.\n"
            "Action: semantic_search\n"
            'Action Input: {"query": "door"}'
        )
        parsed = parse_react_output(text)
        assert parsed["thought"] == "I need to search memory."
        assert parsed["action"] == "semantic_search"
        assert parsed["action_input"] == {"query": "door"}
        assert parsed["final_answer"] is None

    def test_final_answer(self):
        text = (
            "Thought: I now have enough information.\n"
            "Final Answer: The door is at position (3, 5)."
        )
        parsed = parse_react_output(text)
        assert parsed["thought"] == "I now have enough information."
        assert parsed["final_answer"] == "The door is at position (3, 5)."
        assert parsed["action"] is None

    def test_no_thought(self):
        text = "Action: body_status\nAction Input: {}"
        parsed = parse_react_output(text)
        assert parsed["action"] == "body_status"
        assert parsed["action_input"] == {}

    def test_complex_action_input(self):
        text = (
            "Thought: Searching near the agent.\n"
            "Action: spatial_query\n"
            'Action Input: {"x": 3.0, "y": 5.0, "radius": 2.0}'
        )
        parsed = parse_react_output(text)
        assert parsed["action_input"] == {"x": 3.0, "y": 5.0, "radius": 2.0}

    def test_malformed_json(self):
        text = (
            "Thought: trying\n"
            "Action: semantic_search\n"
            "Action Input: {not valid json}"
        )
        parsed = parse_react_output(text)
        assert parsed["action"] == "semantic_search"
        # Should not crash, returns empty dict
        assert parsed["action_input"] == {}

    def test_multiline_final_answer(self):
        text = (
            "Thought: done\n"
            "Final Answer: I visited several rooms:\n"
            "- A kitchen\n"
            "- A hallway"
        )
        parsed = parse_react_output(text)
        assert "kitchen" in parsed["final_answer"]
        assert "hallway" in parsed["final_answer"]

    def test_nested_json_action_input(self):
        text = (
            "Thought: need spatial search\n"
            "Action: spatial_query\n"
            'Action Input: {"x": 3.0, "y": 5.0, "filter": {"layer": "description"}}'
        )
        parsed = parse_react_output(text)
        assert parsed["action_input"] == {
            "x": 3.0,
            "y": 5.0,
            "filter": {"layer": "description"},
        }

    def test_no_action_input(self):
        text = "Thought: checking\nAction: body_status"
        parsed = parse_react_output(text)
        assert parsed["action"] == "body_status"
        assert parsed["action_input"] is None


class TestBuildSystemPrompt:
    def test_builds_from_tool_defs(self):
        from harness.agent.prompts import build_system_prompt

        tool_defs = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search term"},
                    },
                    "required": ["query"],
                },
            }
        ]
        prompt = build_system_prompt(tool_defs)
        assert "test_tool" in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Final Answer:" in prompt


@pytest.mark.ollama
class TestReactAgentIntegration:
    def test_agent_runs_query(self):
        from emem import SpatioTemporalMemory

        from harness.agent.react_agent import ReactAgent
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider
        from harness.providers.ollama_llm import OllamaLLMClient

        embedder = OllamaEmbeddingProvider()
        llm = OllamaLLMClient()

        with SpatioTemporalMemory(
            db_path=":memory:",
            embedding_provider=embedder,
            llm_client=llm,
        ) as mem:
            mem.start_episode("test")
            mem.add("I see a red door ahead", x=3.0, y=5.0, layer_name="description")
            mem.add_body_state("battery: 85%", layer_name="battery")
            mem.end_episode(consolidate=False)

            agent = ReactAgent(mem, max_steps=3)
            result = agent.run("What's my battery level?")

            assert result.answer
            assert len(result.steps) > 0


class _FakeMem:
    """Minimal memory stub for unit-testing the tool-call loop."""

    def __init__(self, tool_defs, responses):
        self._tool_defs = tool_defs
        self._responses = dict(responses)
        self.calls = []

    def get_tool_definitions(self):
        return self._tool_defs

    def dispatch_tool_call(self, name, args):
        self.calls.append((name, args))
        return self._responses.get(name, f"(no response for {name})")


class TestNativeToolCallAgent:
    """A8: Ollama native tool-calling agent."""

    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "search memory semantically",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "temporal_query",
                "description": "search by time",
                "parameters": {
                    "type": "object",
                    "properties": {"t_start": {"type": "number"}},
                    "required": ["t_start"],
                },
            },
        },
    ]

    def _make_agent(self, monkeypatch, scripted_responses, mem_responses=None):
        """Install a monkeypatch on ``post_json`` that returns scripted
        assistant messages in order, and return the constructed agent."""
        from harness.agent import react_agent as react_mod

        mem = _FakeMem(self._TOOLS, mem_responses or {})

        captured_bodies = []
        calls = iter(scripted_responses)

        def fake_post_json(url, body, **_kwargs):
            captured_bodies.append(body)
            return {"message": next(calls)}

        monkeypatch.setattr(react_mod, "post_json", fake_post_json)

        agent = react_mod.NativeToolCallAgent(
            mem,
            model="test-model",
            base_url="http://localhost:11434",
            max_steps=4,
            seed=0,
        )
        return agent, mem, captured_bodies

    def test_dispatches_tool_calls(self, monkeypatch):
        scripted = [
            {
                "role": "assistant",
                "content": "checking memory",
                "tool_calls": [
                    {
                        "function": {
                            "name": "semantic_search",
                            "arguments": {"query": "battery"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The battery is at 85%.",
                "tool_calls": [],
            },
        ]
        agent, mem, _ = self._make_agent(
            monkeypatch, scripted, mem_responses={"semantic_search": "battery: 85%"}
        )
        result = agent.run("What's my battery level?")

        assert result.answer == "The battery is at 85%."
        assert result.tools_used == ["semantic_search"]
        assert mem.calls == [("semantic_search", {"query": "battery"})]

    def test_sends_tools_field_in_body(self, monkeypatch):
        scripted = [{"role": "assistant", "content": "done", "tool_calls": []}]
        agent, _mem, bodies = self._make_agent(monkeypatch, scripted)
        agent.run("hello")
        assert bodies, "no HTTP body captured"
        assert "tools" in bodies[0]
        assert bodies[0]["tools"] == self._TOOLS

    def test_seed_threaded_into_options(self, monkeypatch):
        scripted = [{"role": "assistant", "content": "done", "tool_calls": []}]
        agent, _mem, bodies = self._make_agent(monkeypatch, scripted)
        agent.run("hi")
        assert bodies[0].get("options", {}).get("seed") == 0

    def test_no_tool_calls_returns_content_directly(self, monkeypatch):
        scripted = [
            {
                "role": "assistant",
                "content": "I don't need to search; it's blue.",
                "tool_calls": [],
            }
        ]
        agent, mem, _ = self._make_agent(monkeypatch, scripted)
        result = agent.run("What colour is the sky?")
        assert result.answer == "I don't need to search; it's blue."
        assert result.tools_used == []
        assert mem.calls == []

    def test_multi_step_tool_chain(self, monkeypatch):
        scripted = [
            {
                "role": "assistant",
                "content": "looking up",
                "tool_calls": [
                    {
                        "function": {
                            "name": "semantic_search",
                            "arguments": {"query": "robot"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "now time window",
                "tool_calls": [
                    {
                        "function": {
                            "name": "temporal_query",
                            "arguments": {"t_start": 0.0},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "final: ok",
                "tool_calls": [],
            },
        ]
        agent, _, _ = self._make_agent(monkeypatch, scripted)
        result = agent.run("two-step question")
        assert result.tools_used == ["semantic_search", "temporal_query"]
        assert result.answer == "final: ok"

    def test_max_steps_cuts_off_runaway_tool_calls(self, monkeypatch):
        runaway = {
            "role": "assistant",
            "content": "still searching",
            "tool_calls": [
                {
                    "function": {
                        "name": "semantic_search",
                        "arguments": {"query": "x"},
                    }
                }
            ],
        }
        scripted = [runaway] * 20  # more than max_steps
        agent, _mem, _ = self._make_agent(monkeypatch, scripted)
        result = agent.run("loop forever")
        assert "Reached max steps" in result.answer
        assert len(result.tools_used) == 4  # max_steps=4

    def test_arguments_as_json_string_is_coerced(self, monkeypatch):
        scripted = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "semantic_search",
                            # Ollama normally returns a dict but an older
                            # build can hand back the JSON as a string.
                            "arguments": '{"query": "chair"}',
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "done", "tool_calls": []},
        ]
        agent, mem, _ = self._make_agent(monkeypatch, scripted)
        agent.run("string-args")
        assert mem.calls == [("semantic_search", {"query": "chair"})]

    def test_think_tag_is_stripped(self, monkeypatch):
        scripted = [
            {
                "role": "assistant",
                "content": "<think>lots of reasoning</think>final",
                "tool_calls": [],
            }
        ]
        agent, _mem, _ = self._make_agent(monkeypatch, scripted)
        result = agent.run("thinking")
        assert "think" not in result.answer.lower()
        assert "final" in result.answer
