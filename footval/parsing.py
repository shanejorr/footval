"""Candidate-output parsing policy and judge-input anonymization.

The parse policy is fixed and deterministic — it is metadata extraction, not
repair: direct ``json.loads``, else the contents of exactly one fenced code
block, else failure. The chosen mode is recorded and surfaced to judges.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL)

# Vendor / model identifiers that must never reach a judge.
_REDACT_RE = re.compile(
    r"\b("
    r"claude[\w.\-]*|anthropic|opus|sonnet|haiku|fable"
    r"|gpt[\w.\-]*|openai|chatgpt"
    r"|gemini[\w.\-]*|google"
    r"|deepseek[\w.\-]*"
    r")\b",
    re.IGNORECASE,
)


def parse_candidate_json(raw_text: str) -> tuple[Any | None, str]:
    """Parse a candidate response. Returns ``(parsed, parse_mode)``.

    ``parse_mode`` is one of ``direct`` / ``fenced`` / ``failed``.
    """
    try:
        return json.loads(raw_text), "direct"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    fences = _FENCE_RE.findall(raw_text or "")
    if len(fences) == 1:
        try:
            return json.loads(fences[0]), "fenced"
        except (json.JSONDecodeError, ValueError):
            pass
    return None, "failed"


def redact(text: str) -> tuple[str, int]:
    """Replace vendor/model-identifying tokens with ``[redacted]``.

    Returns the redacted text and the number of replacements made.
    """
    return _REDACT_RE.subn("[redacted]", text)
