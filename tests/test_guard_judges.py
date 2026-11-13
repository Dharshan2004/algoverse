"""Offline checks for Llama Guard and ShieldGemma message construction and parsing."""
import pytest

from refusal_cues.guard_judges import GUARDS, GuardRunner, guard_messages, parse_guard


def test_public_names_import():
    assert set(GUARDS) == {"llama_guard", "shieldgemma"}
    assert callable(GuardRunner)
    assert GUARDS["llama_guard"] == "meta-llama/Llama-Guard-3-1B"
    assert GUARDS["shieldgemma"] == "google/shieldgemma-2b"


def test_llama_guard_uses_typed_text_blocks():
    turns = guard_messages("request text", "response text", "llama_guard")
    assert turns == [
        {"role": "user", "content": [{"type": "text", "text": "request text"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "response text"}]},
    ]


def test_shieldgemma_and_other_guards_use_strings():
    turns = guard_messages("request text", "response text", "shieldgemma")
    assert turns == [
        {"role": "user", "content": "request text"},
        {"role": "assistant", "content": "response text"},
    ]
    assert guard_messages("request text", "response text", "qwen3guard") == turns


def test_llama_guard_parser():
    assert parse_guard("safe", "llama_guard") == "SAFE"
    assert parse_guard("  safe\n", "llama_guard") == "SAFE"
    assert parse_guard("unsafe", "llama_guard") == "UNSAFE"
    assert parse_guard("unsafe S1", "llama_guard") == "UNSAFE"
    assert parse_guard("unsafe S1, S2", "llama_guard") == "UNSAFE"
    assert parse_guard("unsafe S1,S2", "llama_guard") == "UNSAFE"
    for text in ("SAFE", "safe extra", "unsafe because", "unsafe S", ""):
        assert parse_guard(text, "llama_guard") == "INVALID"


def test_shieldgemma_parser_is_score_based():
    with pytest.raises(ValueError, match="Yes/No"):
        parse_guard("Yes", "shieldgemma")
