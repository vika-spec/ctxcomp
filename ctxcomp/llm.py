"""LLM backends.

All backends use the same text protocol for tools (a <tool_call>{...}</tool_call>
block), because small open-source models handle it more reliably than native
function-calling APIs, and it keeps every backend comparable.

Backends:
- OllamaBackend: local models, e.g. `ollama pull smollm2:1.7b` or `qwen2.5:7b`.
- OpenAICompatBackend: any OpenAI-compatible server (vLLM, llama.cpp server,
  LM Studio, hosted APIs).
- ScriptedLLM: deterministic replay of scripted tool calls. It needs no model,
  so the harness can measure exactly what a context strategy keeps or loses.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .messages import Message, ToolCall
from .tokens import count_message_tokens, count_tokens


@dataclass
class LLMOutput:
    text: str
    prompt_tokens: int
    output_tokens: int


def to_chat(messages: list[Message]) -> list[dict[str, str]]:
    """Map internal messages to plain chat roles. Tool results become user
    messages with a clear header, which small models' chat templates accept."""
    out: list[dict[str, str]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "user", "content": f"[TOOL RESULT: {m.name}]\n{m.content}"})
        else:
            out.append({"role": m.role, "content": m.content})
    # Merge consecutive same-role messages (some templates require alternation and a
    # single system message, so the pinned memory joins the system prompt).
    merged: list[dict[str, str]] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(dict(msg))
    return merged


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], **hints: Any) -> LLMOutput: ...


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract the first tool call from model output, tolerating common formats."""
    candidates = [m.group(1) for m in _TAG_RE.finditer(text)]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
    if not candidates and ("tool_call" in text or '"name"' in text or '"tool"' in text):
        obj = _first_json_object(text)
        if obj:
            candidates.append(obj)
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            return ToolCall(name=name, args=args)
    return None


# ---------------------------------------------------------------------------
# Real backends
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 num_ctx: int = 8192, max_tokens: int = 512, temperature: float = 0.0,
                 timeout: float = 300.0) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        self.num_ctx, self.max_tokens, self.temperature, self.timeout = num_ctx, max_tokens, temperature, timeout

    def generate(self, messages: list[Message], **hints: Any) -> LLMOutput:
        import requests
        resp = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
            "model": self.model,
            "messages": to_chat(messages),
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        })
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return LLMOutput(text,
                         data.get("prompt_eval_count") or sum(count_message_tokens(m.content) for m in messages),
                         data.get("eval_count") or count_tokens(text))


class OpenAICompatBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "none",
                 max_tokens: int = 512, temperature: float = 0.0, timeout: float = 300.0) -> None:
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout

    def generate(self, messages: list[Message], **hints: Any) -> LLMOutput:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions", timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": to_chat(messages),
                  "max_tokens": self.max_tokens, "temperature": self.temperature})
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return LLMOutput(text,
                         usage.get("prompt_tokens") or sum(count_message_tokens(m.content) for m in messages),
                         usage.get("completion_tokens") or count_tokens(text))


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------

class ScriptedLLM(LLMBackend):
    """Replays scripted tool calls for each turn, then a short reply.

    Replay mode isolates the context strategy from model quality: the same
    tool calls happen in the same order for every strategy, so differences in
    what reaches the final prompt are caused by the context manager alone.
    """

    def __init__(self) -> None:
        self._plan: list[ToolCall] = []
        self._reply: str | None = None

    def set_plan(self, tool_calls: list[dict], reply: str | None = None) -> None:
        self._plan = [ToolCall(c["tool"], dict(c.get("args", {}))) for c in tool_calls]
        self._reply = reply

    def generate(self, messages: list[Message], **hints: Any) -> LLMOutput:
        prompt_tokens = sum(count_message_tokens(m.content) for m in messages)
        if self._plan:
            text = self._plan.pop(0).to_text()
        elif self._reply is not None:
            text = self._reply
        else:
            text = self._auto_reply(messages)
        return LLMOutput(text, prompt_tokens, count_tokens(text))

    @staticmethod
    def _auto_reply(messages: list[Message]) -> str:
        last = messages[-1] if messages else None
        if last is None or last.role != "tool":
            return "Got it. What would you like to plan next?"
        try:
            data = json.loads(last.content)
        except json.JSONDecodeError:
            first = last.content.splitlines()[0] if last.content else ""
            return f"Done. {first}"
        items = data.get("results") or data.get("departures") or data.get("forecast") or []
        names = []
        for it in items[:3]:
            if isinstance(it, dict):
                label = it.get("name") or it.get("title") or it.get("airline") or it.get("departs") or it.get("day")
                price = it.get("price_per_night_usd") or it.get("price_usd") or it.get("avg_price_usd") \
                    or it.get("fare_from_usd")
                names.append(f"{label} (${price})" if price is not None else str(label))
        return f"I found {len(items)} options. Top picks: " + "; ".join(names) + "." if names else "Here is what I found."


def make_backend(kind: str, **kwargs: Any) -> LLMBackend:
    kind = kind.lower()
    if kind == "ollama":
        return OllamaBackend(**kwargs)
    if kind in ("openai", "openai_compat", "vllm"):
        return OpenAICompatBackend(**kwargs)
    if kind in ("scripted", "replay"):
        return ScriptedLLM()
    raise ValueError(f"unknown backend '{kind}'")
