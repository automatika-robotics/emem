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
