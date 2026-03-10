from harness.providers.http import strip_think_tags


class TestStripThinkTags:
    def test_no_tags(self):
        assert strip_think_tags("hello world") == "hello world"

    def test_simple_tag(self):
        assert strip_think_tags("<think>reasoning</think>answer") == "answer"

    def test_multiline_tag(self):
        text = "<think>\nlong\nreasoning\n</think>\nthe answer"
        assert strip_think_tags(text) == "the answer"

    def test_only_thinking(self):
        assert strip_think_tags("<think>just thinking</think>") == ""

    def test_content_before_and_after(self):
        text = "prefix <think>middle</think> suffix"
        assert strip_think_tags(text) == "prefix  suffix"

    def test_empty_string(self):
        assert strip_think_tags("") == ""
