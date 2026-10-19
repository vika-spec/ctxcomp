"""Deterministic checks on the final answer (live mode)."""

from __future__ import annotations

import re
from typing import Any


def _hits(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.I)]


def run_answer_check(check: dict[str, Any], answer: str) -> dict[str, Any]:
    kind = check["type"]
    if kind == "forbid":
        hits = _hits(check["patterns"], answer)
        return {"id": check.get("id", kind), "passed": not hits, "detail": {"forbidden_found": hits}}
    if kind == "forbid_unless":
        hits = _hits(check["forbid"], answer)
        excuses = _hits(check["unless"], answer)
        return {"id": check.get("id", kind), "passed": (not hits) or bool(excuses),
                "detail": {"forbidden_found": hits, "mitigations_found": excuses}}
    if kind == "require_any":
        hits = _hits(check["patterns"], answer)
        return {"id": check.get("id", kind), "passed": bool(hits), "detail": {"found": hits}}
    raise ValueError(f"unknown check type {kind}")
