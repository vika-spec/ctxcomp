"""The plug-in point: every context strategy implements this interface.

The agent never touches its own history directly. It reports events
(user message, tool call, tool result, assistant reply) to a ContextManager,
and asks it for the prompt to send to the model. Swapping the ContextManager
is the only change between the baseline and the compressed agent, which keeps
the before-vs-after comparison fair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..messages import Message, ToolCall
from ..tokens import count_message_tokens


class ContextManager(ABC):
    name: str = "base"

    def __init__(self, max_context_tokens: int = 8192, reserve_output_tokens: int = 512) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.events: list[dict[str, Any]] = []   # decisions log (what was kept / dropped / compressed)

    @property
    def prompt_budget(self) -> int:
        return self.max_context_tokens - self.reserve_output_tokens

    # ---- events reported by the agent ------------------------------------
    @abstractmethod
    def on_user(self, text: str, turn: int) -> None: ...

    @abstractmethod
    def on_tool_call(self, call: ToolCall, turn: int) -> None: ...

    @abstractmethod
    def on_tool_result(self, call: ToolCall, result: str, turn: int) -> None: ...

    @abstractmethod
    def on_assistant(self, text: str, turn: int) -> None: ...

    # ---- prompt assembly ---------------------------------------------------
    @abstractmethod
    def build(self, system_prompt: str) -> list[Message]:
        """Return the messages to send. Must fit within prompt_budget."""

    def stats(self) -> dict[str, Any]:
        return {}

    def new_session(self) -> "ContextManager":
        """Start a new session (e.g. the user comes back the next day). The default
        keeps nothing, which is what a stateless agent does. Strategies with
        persistent memory override this to carry their state over."""
        fresh = type(self)(self.max_context_tokens, self.reserve_output_tokens)
        fresh.log(action="new_session", carried_over="nothing")
        return fresh

    def log(self, **event: Any) -> None:
        self.events.append(event)


def messages_tokens(messages: list[Message]) -> int:
    return sum(count_message_tokens(m.content) for m in messages)
