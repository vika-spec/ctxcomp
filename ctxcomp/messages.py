"""Message and event types shared by the agent, context managers and LLM backends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    turn: int = 0                 # conversation turn this message belongs to (0 = system)
    name: str | None = None       # tool name for role="tool"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

    def to_text(self) -> str:
        return "<tool_call>" + json.dumps({"name": self.name, "arguments": self.args}) + "</tool_call>"


@dataclass
class LLMCallRecord:
    """One model invocation, with what the model actually saw."""
    turn: int
    step: int
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    prompt_text: str              # flattened prompt, used for retention / leakage checks
    output_text: str
    prompt_parts: list[dict[str, Any]] = field(default_factory=list)  # per-message summary for inspection


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def summarize_parts(messages: list[Message]) -> list[dict[str, Any]]:
    """Compact description of each prompt message (no full text)."""
    from .tokens import count_message_tokens
    return [{"role": m.role, "name": m.name, "turn": m.turn, "tokens": count_message_tokens(m.content),
             "hash": content_hash(m.content), "kind": m.meta.get("kind"), "src": m.meta.get("src_hash")}
            for m in messages]


def flatten(messages: list[Message]) -> str:
    """Plain-text rendering of a prompt, for inspection and fact checks."""
    parts = []
    for m in messages:
        tag = f"{m.role}:{m.name}" if m.name else m.role
        parts.append(f"[{tag}]\n{m.content}")
    return "\n\n".join(parts)
