"""Travel concierge agent.

The agent loop is deliberately simple and strategy-agnostic: it reports every
event to the ContextManager and asks it for the prompt. The context strategy is
the only thing that changes between the baseline and the compressed agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .context.base import ContextManager
from .llm import LLMBackend, parse_tool_call
from .messages import LLMCallRecord, ToolCall, flatten, summarize_parts
from .tools import ToolRegistry, make_registry

SYSTEM_PROMPT = """You are a meticulous travel concierge helping one user plan a multi-city trip.

Rules:
- Always respect the user's stated preferences and constraints (budget, dietary restrictions and allergies, mobility, pace, fixed appointments). Reference them explicitly when relevant.
- Track the budget continuously: know the cap the user gave and what has been spent so far.
- Proactively flag conflicts between a request and earlier constraints before acting on it.
- When the user changes their plans, stop using anything they cancelled.
- Use tools to look things up. Never invent prices or availability.

Tools:
{tools}

To call a tool, reply with ONLY this block and nothing else:
<tool_call>{{"name": "<tool name>", "arguments": {{...}}}}</tool_call>

You will receive the result in a message starting with [TOOL RESULT]. Call tools one at a time.
When you have what you need, reply to the user in plain text (no tool_call block)."""


@dataclass
class TurnResult:
    turn: int
    user: str
    final_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    hit_step_limit: bool = False


class TravelAgent:
    def __init__(self, llm: LLMBackend, context: ContextManager, tools: ToolRegistry | None = None,
                 max_steps: int = 5, system_prompt: str | None = None) -> None:
        self.llm = llm
        self.context = context
        self.tools = tools or make_registry()
        self.max_steps = max_steps
        self.system_prompt = (system_prompt or SYSTEM_PROMPT).format(tools=self.tools.render())
        self.turn = 0

    def chat(self, user_text: str) -> TurnResult:
        self.turn += 1
        turn = self.turn
        result = TurnResult(turn=turn, user=user_text, final_text="")
        self.context.on_user(user_text, turn)

        for step in range(self.max_steps):
            messages = self.context.build(self.system_prompt)
            t0 = time.perf_counter()
            out = self.llm.generate(messages, turn=turn, step=step)
            latency = time.perf_counter() - t0
            result.llm_calls.append(LLMCallRecord(
                turn=turn, step=step, prompt_tokens=out.prompt_tokens, output_tokens=out.output_tokens,
                latency_s=latency, prompt_text=flatten(messages), output_text=out.text,
                prompt_parts=summarize_parts(messages)))

            call = parse_tool_call(out.text)
            if call is None:
                result.final_text = out.text.strip()
                self.context.on_assistant(result.final_text, turn)
                return result

            result.tool_calls.append(call)
            self.context.on_tool_call(call, turn)
            tool_output = self.tools.call(call.name, call.args)
            self.context.on_tool_result(call, tool_output, turn)

        result.hit_step_limit = True
        result.final_text = "(stopped: reached the tool-call limit for this turn)"
        self.context.on_assistant(result.final_text, turn)
        return result
