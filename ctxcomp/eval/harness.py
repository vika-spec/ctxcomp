"""Replay scripted conversations through an agent and score the result.

Two modes:
- replay: ScriptedLLM issues the scenario's tool calls. No model needed.
  Scores what the context strategy keeps in the final prompt: retention of
  critical facts and leakage of stale ones. Fully deterministic.
- live: a real model chooses its own tool calls and writes the answer.
  Additionally scores the final answer with deterministic checks.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..agent import TravelAgent
from ..context.base import ContextManager
from ..llm import LLMBackend, ScriptedLLM
from ..tools import make_registry
from .checks import run_answer_check
from .scenario import Scenario


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    layer: str
    strategy: str
    mode: str
    max_context_tokens: int
    turns: int
    # context-level (both modes)
    retained: dict[str, bool]
    stale_leaked: dict[str, bool]
    retention_rate: float
    context_pass: bool
    # answer-level (live mode only)
    answer_checks: list[dict[str, Any]] | None
    answer_pass: bool | None
    final_answer: str
    # efficiency
    llm_calls: int
    tool_calls: int
    total_prompt_tokens: int
    mean_prompt_tokens: float
    peak_prompt_tokens: int
    final_prompt_tokens: int
    prompt_tokens_per_turn: list[int]
    total_latency_s: float
    p95_call_latency_s: float
    wall_time_s: float
    context_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def task_pass(self) -> bool:
        """Live: the answer passed. Replay: the context kept what the answer needs."""
        return self.answer_pass if self.answer_pass is not None else self.context_pass

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_pass"] = self.task_pass
        return d


def run_scenario(
    scenario: Scenario,
    make_context: Callable[[], ContextManager],
    backend: LLMBackend | None = None,
    max_steps: int = 5,
) -> ScenarioResult:
    mode = "live" if backend is not None else "replay"
    llm = backend if backend is not None else ScriptedLLM()
    context = make_context()
    agent = TravelAgent(llm, context, make_registry(), max_steps=max_steps)

    errors: list[str] = []
    all_calls = []
    per_turn_final_prompt: list[int] = []
    last = None
    t_start = time.perf_counter()
    for turn in scenario.turns:
        if turn.new_session:
            context = context.new_session()
            agent.context = context
        if isinstance(llm, ScriptedLLM):
            llm.set_plan(turn.tools, turn.reply)
        try:
            last = agent.chat(turn.user)
        except Exception as e:  # keep evaluating other scenarios
            errors.append(f"turn {agent.turn}: {type(e).__name__}: {e}")
            break
        all_calls.extend(last.llm_calls)
        per_turn_final_prompt.append(last.llm_calls[-1].prompt_tokens if last.llm_calls else 0)
    wall = time.perf_counter() - t_start

    final_prompt = last.llm_calls[-1].prompt_text if last and last.llm_calls else ""
    retained = {c.id: c.found_in(final_prompt) for c in scenario.retained}
    leaked = {c.id: c.found_in(final_prompt) for c in scenario.stale}
    retention_rate = (sum(retained.values()) / len(retained)) if retained else 1.0
    context_pass = all(retained.values()) and not any(leaked.values()) and not errors

    final_answer = last.final_text if last else ""
    answer_checks = None
    answer_pass = None
    if mode == "live":
        answer_checks = [run_answer_check(c, final_answer) for c in scenario.answer]
        answer_pass = all(c["passed"] for c in answer_checks) and not errors

    prompt_tokens = [c.prompt_tokens for c in all_calls] or [0]
    latencies = sorted(c.latency_s for c in all_calls) or [0.0]
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]

    return ScenarioResult(
        scenario_id=scenario.id, title=scenario.title, layer=scenario.layer,
        strategy=context.name, mode=mode, max_context_tokens=context.max_context_tokens,
        turns=len(per_turn_final_prompt),
        retained=retained, stale_leaked=leaked, retention_rate=retention_rate, context_pass=context_pass,
        answer_checks=answer_checks, answer_pass=answer_pass, final_answer=final_answer,
        llm_calls=len(all_calls), tool_calls=sum(1 for c in all_calls if "<tool_call>" in c.output_text),
        total_prompt_tokens=sum(prompt_tokens), mean_prompt_tokens=statistics.fmean(prompt_tokens),
        peak_prompt_tokens=max(prompt_tokens),
        final_prompt_tokens=all_calls[-1].prompt_tokens if all_calls else 0,
        prompt_tokens_per_turn=per_turn_final_prompt,
        total_latency_s=sum(c.latency_s for c in all_calls), p95_call_latency_s=p95, wall_time_s=wall,
        context_stats=context.stats(), errors=errors,
    )
