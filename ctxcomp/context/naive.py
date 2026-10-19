"""Baseline: full-history stuffing with sliding-window truncation.

This is what most agents do by default: append everything (user turns, raw
tool outputs, replies) and, when the prompt no longer fits, drop the oldest
messages. It has no idea what is important, so it forgets a shellfish allergy
just as readily as a stale weather report.
"""

from __future__ import annotations

from ..messages import Message, ToolCall
from ..tokens import count_message_tokens
from .base import ContextManager


class NaiveContext(ContextManager):
    name = "naive"

    def __init__(self, max_context_tokens: int = 8192, reserve_output_tokens: int = 512) -> None:
        super().__init__(max_context_tokens, reserve_output_tokens)
        self.history: list[Message] = []
        self.dropped_messages = 0
        self.dropped_tokens = 0
        self.overflow_builds = 0

    def on_user(self, text: str, turn: int) -> None:
        self.history.append(Message("user", text, turn))

    def on_tool_call(self, call: ToolCall, turn: int) -> None:
        self.history.append(Message("assistant", call.to_text(), turn, meta={"tool_call": True}))

    def on_tool_result(self, call: ToolCall, result: str, turn: int) -> None:
        self.history.append(Message("tool", result, turn, name=call.name))

    def on_assistant(self, text: str, turn: int) -> None:
        self.history.append(Message("assistant", text, turn))

    def build(self, system_prompt: str) -> list[Message]:
        system = Message("system", system_prompt, 0)
        budget = self.prompt_budget - count_message_tokens(system_prompt)
        kept: list[Message] = []
        used = 0
        # Walk backwards from the newest message, keep what fits.
        for m in reversed(self.history):
            t = count_message_tokens(m.content)
            if used + t > budget:
                break
            kept.append(m)
            used += t
        kept.reverse()
        if not kept and self.history:
            # The newest message alone is larger than the budget: keep its tail.
            last = self.history[-1]
            max_chars = max(0, budget - 8) * 4
            kept = [Message(last.role, last.content[-max_chars:], last.turn, last.name, dict(last.meta))]
        n_drop = len(self.history) - len(kept)
        dropped = self.history[:n_drop]
        self.dropped_messages = n_drop
        self.dropped_tokens = sum(count_message_tokens(m.content) for m in dropped)
        if n_drop:
            self.overflow_builds += 1
            self.log(action="truncate", dropped_messages=n_drop, oldest_kept_turn=kept[0].turn if kept else None)
        return [system] + kept

    def stats(self) -> dict:
        return {
            "history_messages": len(self.history),
            "history_tokens": sum(count_message_tokens(m.content) for m in self.history),
            "dropped_messages_last_build": self.dropped_messages,
            "dropped_tokens_last_build": self.dropped_tokens,
            "builds_with_truncation": self.overflow_builds,
        }
