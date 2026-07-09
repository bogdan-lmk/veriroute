"""Sandbox execution and PoT plumbing (no model needed — LocalLLM is mocked)."""
from agent import pot
from agent.sandbox import extract_code, run_python


class TestExtractCode:
    def test_fenced_python(self):
        assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1"

    def test_fenced_bare(self):
        assert extract_code("```\nprint(2)\n```") == "print(2)"

    def test_raw_fallback(self):
        assert extract_code("print(3)") == "print(3)"


class TestRunPython:
    def test_happy_path(self):
        ok, out = run_python("print(6 * 7)")
        assert ok and out == "42"

    def test_nonzero_exit(self):
        ok, out = run_python("raise ValueError('boom')")
        assert not ok and "boom" in out

    def test_timeout(self):
        ok, out = run_python("while True: pass", timeout_s=1.0)
        assert not ok and out == "timeout"

    def test_failed_assert_is_failure(self):
        ok, _ = run_python("def f(): return 1\nassert f() == 2")
        assert not ok


class FakeLLM:
    """Stands in for LocalLLM: returns queued replies."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, prompt, max_tokens, timeout_s):  # noqa: ARG002
        return self.replies.pop(0)


class TestMathPot:
    PROMPT = "A store has 240 items. It sells 15% on Monday and 60 more. How many remain?"

    def test_correct_program_yields_answer(self):
        llm = FakeLLM([
            "```python\ndef solve():\n    return 240 - int(240*0.15) - 60\nprint(solve())\n```"
        ])
        assert pot.math_pot(llm, self.PROMPT, 10.0) == "144"

    def test_missing_print_is_appended(self):
        llm = FakeLLM(["```python\ndef solve():\n    return 5\n```"])
        assert pot.math_pot(llm, self.PROMPT, 10.0) == "5"

    def test_crash_rejected(self):
        llm = FakeLLM(["```python\nprint(unknown_var)\n```"])
        assert pot.math_pot(llm, self.PROMPT, 10.0) is None

    def test_prose_output_rejected(self):
        llm = FakeLLM(["```python\nprint('the answer is many')\n```"])
        assert pot.math_pot(llm, self.PROMPT, 10.0) is None


class TestCodegenSelftested:
    PROMPT = "Write a Python function second_largest(nums) returning the second largest distinct number."

    GOOD_BLOCK = (
        "```python\ndef second_largest(nums):\n"
        "    d = sorted(set(nums))\n    return d[-2]\n\n"
        "assert second_largest([5,1,5,3]) == 3\n"
        "assert second_largest([1,2]) == 1\n"
        "assert second_largest([9,9,8]) == 8\n```"
    )

    def test_passing_tests_ships_code_without_asserts(self):
        llm = FakeLLM([self.GOOD_BLOCK])
        answer = pot.codegen_selftested(llm, self.PROMPT, 20.0)
        assert answer is not None
        assert "def second_largest" in answer
        assert "assert" not in answer

    def test_failing_tests_reject(self):
        llm = FakeLLM([
            "```python\ndef second_largest(nums):\n    return max(nums)\n\n"
            "assert second_largest([5,1,3]) == 3\n```",
        ])
        assert pot.codegen_selftested(llm, self.PROMPT, 20.0) is None

    def test_no_function_rejected(self):
        llm = FakeLLM(["just some text without code"])
        assert pot.codegen_selftested(llm, self.PROMPT, 20.0) is None

    def test_no_asserts_rejected(self):
        llm = FakeLLM([
            "```python\ndef second_largest(nums):\n"
            "    return sorted(set(nums))[-2]\n```",
        ])
        assert pot.codegen_selftested(llm, self.PROMPT, 20.0) is None
