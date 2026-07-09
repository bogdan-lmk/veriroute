"""Cheap deterministic verifiers for local answers."""
from agent import verifiers


class TestSentiment:
    PROMPT = "Classify the sentiment of this review: great battery, bad screen."

    def test_accepts_common_label(self):
        assert verifiers.verify_sentiment(self.PROMPT, "Mixed — praise and criticism.")

    def test_rejects_empty_and_labelless(self):
        assert not verifiers.verify_sentiment(self.PROMPT, "")
        assert not verifiers.verify_sentiment(self.PROMPT, "It is about a phone.")

    def test_enumerated_labels_from_prompt(self):
        prompt = "Classify this text as: happy, sad, or angry. Text: ..."
        assert verifiers.verify_sentiment(prompt, "sad")
        assert not verifiers.verify_sentiment(prompt, "positive")

    def test_rejects_essay(self):
        assert not verifiers.verify_sentiment(self.PROMPT, "positive " * 120)


class TestNer:
    PROMPT = "Extract all named entities and their types from: ..."

    def test_accepts_typed_extraction(self):
        assert verifiers.verify_ner(
            self.PROMPT, "Maria Sanchez — PERSON; Berlin — LOCATION"
        )

    def test_accepts_table_shape(self):
        assert verifiers.verify_ner(
            self.PROMPT, "| Maria Sanchez | Person |\n| Berlin | Location |"
        )

    def test_rejects_refusal(self):
        assert not verifiers.verify_ner(self.PROMPT, "no entities were found here")
        assert not verifiers.verify_ner(self.PROMPT, "")


class TestSummarization:
    ONE = "Summarize the following in exactly one sentence: ..."

    def test_accepts_single_sentence(self):
        assert verifiers.verify_summarization(
            self.ONE, "The platform coordinates agents through one pipeline."
        )

    def test_rejects_two_sentences(self):
        assert not verifiers.verify_summarization(
            self.ONE, "It coordinates agents. It also verifies results."
        )

    def test_rejects_missing_terminal_punctuation(self):
        assert not verifiers.verify_summarization(
            self.ONE, "The platform coordinates agents through one pipeline"
        )

    def test_generic_summary_must_compress(self):
        prompt = "Summarize: " + "long text " * 50
        assert verifiers.verify_summarization(prompt, "Short recap of the text.")
        assert not verifiers.verify_summarization(prompt, "word " * 200)
