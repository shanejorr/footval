import json

from footbench.parsing import parse_candidate_json, redact


def test_direct_parse():
    parsed, mode = parse_candidate_json('{"a": 1}')
    assert parsed == {"a": 1}
    assert mode == "direct"


def test_single_fenced_block():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
    parsed, mode = parse_candidate_json(raw)
    assert parsed == {"a": 1}
    assert mode == "fenced"


def test_bare_fence_block():
    raw = '```\n{"a": 1}\n```'
    parsed, mode = parse_candidate_json(raw)
    assert parsed == {"a": 1}
    assert mode == "fenced"


def test_two_fenced_blocks_fail():
    raw = '```json\n{"a": 1}\n```\nand\n```json\n{"b": 2}\n```'
    parsed, mode = parse_candidate_json(raw)
    assert parsed is None
    assert mode == "failed"


def test_prose_wrapped_json_fails():
    parsed, mode = parse_candidate_json('The answer is {"a": 1} as shown.')
    assert parsed is None
    assert mode == "failed"


def test_truncated_json_fails():
    full = json.dumps({"a": list(range(100))})
    parsed, mode = parse_candidate_json(full[:50])
    assert parsed is None
    assert mode == "failed"


def test_redaction_replaces_vendor_names_case_insensitively():
    text = "I am Claude, by ANTHROPIC, unlike GPT-5.5-pro or Gemini-3.1 or DeepSeek."
    redacted, count = redact(text)
    assert count >= 5
    lowered = redacted.lower()
    for token in ("claude", "anthropic", "gpt", "gemini", "deepseek"):
        assert token not in lowered


def test_redaction_leaves_football_text_alone():
    text = "Go for it on 4th-and-2; play-action off outside zone beats Cover 3."
    redacted, count = redact(text)
    assert count == 0
    assert redacted == text
