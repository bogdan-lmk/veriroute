"""ALLOWED_MODELS parsing degradations and token-oriented ranking."""
from agent.model_ranking import (
    is_thinking_likely,
    parse_allowed_models,
    rank_models,
    size_billions,
    supports_reasoning_effort,
)

PREFIX = "accounts/fireworks/models/"


class TestParseAllowedModels:
    def test_none_and_empty(self):
        assert parse_allowed_models(None) == []
        assert parse_allowed_models("") == []
        assert parse_allowed_models("   ") == []

    def test_garbage_separators(self):
        assert parse_allowed_models(" , ,, ") == []

    def test_whitespace_and_trailing_commas(self):
        raw = f" {PREFIX}gpt-oss-20b , {PREFIX}kimi-k2p6 ,"
        assert parse_allowed_models(raw) == [
            f"{PREFIX}gpt-oss-20b",
            f"{PREFIX}kimi-k2p6",
        ]

    def test_duplicates_dropped(self):
        raw = f"{PREFIX}a,{PREFIX}a,{PREFIX}b"
        assert parse_allowed_models(raw) == [f"{PREFIX}a", f"{PREFIX}b"]


class TestSizeParsing:
    def test_plain_and_fractional(self):
        assert size_billions(f"{PREFIX}llama-v3p2-3b-instruct") == 3.0
        assert size_billions(f"{PREFIX}qwen2p5-1p5b-instruct") == 1.5
        assert size_billions(f"{PREFIX}gpt-oss-120b") == 120.0

    def test_no_size(self):
        assert size_billions(f"{PREFIX}deepseek-v4-flash") is None


class TestThinkingDetection:
    def test_known_thinking_families(self):
        assert is_thinking_likely(f"{PREFIX}deepseek-v4-flash")
        assert is_thinking_likely(f"{PREFIX}kimi-k2p6")
        assert is_thinking_likely(f"{PREFIX}glm-5p1")
        assert is_thinking_likely(f"{PREFIX}qwen3-30b-thinking")

    def test_classic_instruct_not_thinking(self):
        assert not is_thinking_likely(f"{PREFIX}llama-v3p1-8b-instruct")
        assert not is_thinking_likely(f"{PREFIX}qwen2p5-7b-instruct")

    def test_gpt_oss_is_effort_controllable(self):
        assert supports_reasoning_effort(f"{PREFIX}gpt-oss-20b")
        assert not is_thinking_likely(f"{PREFIX}gpt-oss-20b")


class TestRanking:
    def test_output_is_permutation_of_input(self):
        models = [f"{PREFIX}whatever-weird-name", f"{PREFIX}another$one", ""]
        models = [m for m in models if m]
        assert sorted(rank_models(models)) == sorted(models)

    def test_non_thinking_before_thinking(self):
        ranked = rank_models([
            f"{PREFIX}deepseek-v4-flash",
            f"{PREFIX}llama-v3p1-8b-instruct",
        ])
        assert ranked[0].endswith("llama-v3p1-8b-instruct")

    def test_larger_non_thinking_first_for_accuracy(self):
        ranked = rank_models([
            f"{PREFIX}gpt-oss-20b",
            f"{PREFIX}gpt-oss-120b",
        ])
        assert ranked[0].endswith("gpt-oss-120b")

    def test_cheap_thinking_variant_before_premium(self):
        ranked = rank_models([
            f"{PREFIX}kimi-k2p6",
            f"{PREFIX}deepseek-v4-flash",
        ])
        assert ranked[0].endswith("deepseek-v4-flash")

    def test_realistic_full_ordering(self):
        ranked = rank_models([
            f"{PREFIX}kimi-k2p6",
            f"{PREFIX}deepseek-v4-flash",
            f"{PREFIX}gpt-oss-20b",
            f"{PREFIX}gpt-oss-120b",
            f"{PREFIX}llama-v3p2-3b-instruct",
        ])
        names = [m.rsplit("/", 1)[-1] for m in ranked]
        assert names[0] == "gpt-oss-120b"          # non-thinking, largest
        assert names[1] == "gpt-oss-20b"           # non-thinking
        assert names[2] == "llama-v3p2-3b-instruct"  # non-thinking, small
        assert names[3] == "deepseek-v4-flash"     # thinking but cheap variant
        assert names[4] == "kimi-k2p6"             # thinking premium

    def test_unknown_names_do_not_crash(self):
        junk = ["x", "a/b/c", "MODEL", "123", "-,-"]
        assert sorted(rank_models(junk)) == sorted(junk)
