"""Helpers for parsing LLM outputs (e.g. JSON inside markdown)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json(text: str) -> str:
    """Strip markdown code blocks and return inner string for JSON parsing."""
    if not text or not isinstance(text, str):
        return "{}"
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_json_safe(text: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse JSON from LLM output; strip code blocks first. Return default on failure."""
    default = default or {}
    try:
        raw = extract_json(text)
        if not raw:
            return default
        out = json.loads(raw)
        return out if isinstance(out, dict) else default
    except Exception:
        return default
