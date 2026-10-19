"""Token counting.

The default counter is a fast character heuristic (about 4 chars per token for
English/JSON), which is close enough for budgeting experiments. For exact
numbers, plug in the target model's tokenizer:

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    set_token_counter(lambda s: len(tok.encode(s, add_special_tokens=False)))
"""

from __future__ import annotations

import math
from typing import Callable

# Per-message overhead for chat templates (role markers, separators).
MESSAGE_OVERHEAD = 4

_counter: Callable[[str], int] | None = None


def _heuristic(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """Install a custom tokenizer function, or None to restore the heuristic."""
    global _counter
    _counter = fn


def count_tokens(text: str) -> int:
    return (_counter or _heuristic)(text)


def count_message_tokens(content: str) -> int:
    return count_tokens(content) + MESSAGE_OVERHEAD
