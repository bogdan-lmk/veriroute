"""Deterministic classifier: practice tasks are the ground truth."""
from agent.classifier import ESCALATE, classify


class TestPracticeTasks:
    """The 8 official practice tasks, one per category."""

    def test_factual(self):
        assert classify(
            "What is the capital of Australia, and what body of water is it near?"
        ) == "factual"

    def test_math(self):
        assert classify(
            "A store has 240 items. It sells 15% on Monday and 60 more on "
            "Tuesday. How many items remain?"
        ) == "math"

    def test_sentiment(self):
        assert classify(
            "Classify the sentiment of this review: The battery life is "
            "great, but the screen scratches too easily."
        ) == "sentiment"

    def test_summarization(self):
        assert classify(
            "Summarize the following in exactly one sentence: The new "
            "platform coordinates agents through an event-driven pipeline."
        ) == "summarization"

    def test_ner(self):
        assert classify(
            "Extract all named entities and their types from: Maria Sanchez "
            "joined Fireworks AI in Berlin last March."
        ) == "ner"

    def test_code_debug(self):
        assert classify(
            "This function should return the max of a list but has a bug: "
            "def get_max(nums): return nums[0]. Find and fix it."
        ) == "code_debug"

    def test_logic(self):
        assert classify(
            "Three friends, Sam, Jo, and Lee, each own a different pet: cat, "
            "dog, bird. Sam does not own the bird. Jo owns the dog. Who owns "
            "the cat?"
        ) == "logic"

    def test_code_gen(self):
        assert classify(
            "Write a Python function that returns the second-largest number "
            "in a list, handling duplicates correctly."
        ) == "code_gen"


class TestConservativeFallbacks:
    def test_empty_prompt(self):
        assert classify("") == ESCALATE
        assert classify("   ") == ESCALATE

    def test_non_english_escalates(self):
        assert classify(
            "Классифицируй тональность этого отзыва: батарея отличная, но "
            "экран легко царапается."
        ) == ESCALATE

    def test_multi_part_escalates(self):
        assert classify(
            "Summarize this review in one sentence and classify its "
            "sentiment: The battery is great but the screen scratches."
        ) == ESCALATE

    def test_keyword_spoof_multiple_categories(self):
        assert classify(
            "Extract all named entities and summarize the passage: ..."
        ) == ESCALATE

    def test_unmatched_statement_escalates(self):
        assert classify(
            "Discuss the philosophical implications of digital consciousness "
            "in modern society and its long-term trajectory."
        ) == ESCALATE

    def test_short_question_is_factual(self):
        assert classify("What year did the Berlin Wall fall?") == "factual"


class TestDisambiguation:
    def test_write_function_with_bug_context_is_debug(self):
        assert classify(
            "There is a bug in this function, fix it: def f(x): return x[0] "
            "so that it returns the max."
        ) == "code_debug"

    def test_math_with_each_is_math_not_logic(self):
        assert classify(
            "Three friends each have 12 apples. They give away 5 apples "
            "total. How many apples remain altogether?"
        ) == "math"

    def test_fix_without_code_not_debug(self):
        # "fix it" without any code signal should not classify as debug.
        result = classify("My printer is broken, how do I fix it?")
        assert result in (ESCALATE, "factual")
