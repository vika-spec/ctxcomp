"""Per-turn traces of what an agent's context strategy keeps, for the inspector UI.

A trace records, for every turn: the full conversation so far (every message the
agent ever produced or received), which of those messages were in the prompt when
the model answered, any synthetic memory the strategy injected instead, and the
status of each critical fact (in context, evicted, or not stated yet).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..agent import TravelAgent
from ..context.base import ContextManager
from ..context.naive import NaiveContext
from ..eval.checks import run_answer_check
from ..eval.scenario import FactCheck, Scenario
from ..llm import LLMBackend, ScriptedLLM, parse_tool_call
from ..messages import Message, ToolCall, content_hash
from ..tokens import count_message_tokens
from ..tools import make_registry

UNLIMITED = 1_000_000


class TeeContext(ContextManager):
    """Forwards events to the real strategy and to an unlimited shadow history,
    so the inspector can compare 'everything said so far' with 'what the model saw'."""

    def __init__(self, primary: ContextManager) -> None:
        super().__init__(primary.max_context_tokens, primary.reserve_output_tokens)
        self.primary = primary
        self.shadow = NaiveContext(max_context_tokens=10**9)
        self.name = primary.name

    def on_user(self, text, turn):
        self.primary.on_user(text, turn); self.shadow.on_user(text, turn)

    def on_tool_call(self, call, turn):
        self.primary.on_tool_call(call, turn); self.shadow.on_tool_call(call, turn)

    def on_tool_result(self, call, result, turn):
        self.primary.on_tool_result(call, result, turn); self.shadow.on_tool_result(call, result, turn)

    def on_assistant(self, text, turn):
        self.primary.on_assistant(text, turn); self.shadow.on_assistant(text, turn)

    def build(self, system_prompt):
        return self.primary.build(system_prompt)

    def new_session(self):
        """The strategy starts a new session; the shadow keeps the full record for the UI."""
        self.primary = self.primary.new_session()
        self.name = self.primary.name
        return self

    def stats(self):
        return self.primary.stats()

    @property
    def events(self):  # decision log of the real strategy
        return self.primary.events

    @events.setter
    def events(self, value):
        pass


def _preview(m: Message) -> str:
    if m.role == "assistant" and m.meta.get("tool_call"):
        call = parse_tool_call(m.content)
        if call:
            args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
            return f"calls {call.name}({args})"
    if m.role == "tool":
        try:
            data = json.loads(m.content)
            bits = [str(data.get(k)) for k in ("query", "location", "route") if data.get(k)]
            n = data.get("result_count") or len(data.get("results") or data.get("departures")
                                                 or data.get("forecast") or [])
            return f"{m.name}: {' / '.join(bits)} ({n} items)".strip()
        except (json.JSONDecodeError, AttributeError):
            return f"{m.name}: " + m.content.splitlines()[0][:100]
    text = re.sub(r"\s+", " ", m.content).strip()
    return text[:140] + ("..." if len(text) > 140 else "")


def _label(strategy: str, ctx: int) -> str:
    if ctx >= UNLIMITED:
        return f"{strategy} · unlimited"
    return f"{strategy} · {ctx // 1024}K" if ctx % 1024 == 0 else f"{strategy} · {ctx}"


class TraceRecorder:
    """Drives one agent conversation and records a trace turn by turn.
    Used for whole-scenario replays and for live chat sessions alike."""

    def __init__(self, strategy: ContextManager, backend: LLMBackend | None,
                 facts: list[FactCheck] | None = None, stale: list[FactCheck] | None = None,
                 max_steps: int = 5) -> None:
        self.ctx = TeeContext(strategy)
        self.llm = backend or ScriptedLLM()
        self.agent = TravelAgent(self.llm, self.ctx, make_registry(), max_steps=max_steps)
        self.facts = facts or []
        self.stale = stale or []
        self.messages: list[dict[str, Any]] = []   # every raw message, in order
        self.turns: list[dict[str, Any]] = []
        self._fact_source: dict[str, int] = {}
        self.session = 1

    @property
    def label(self) -> str:
        return _label(self.ctx.name, self.ctx.max_context_tokens)

    def _sync_messages(self) -> None:
        hist = self.ctx.shadow.history
        for m in hist[len(self.messages):]:
            idx = len(self.messages)
            self.messages.append({
                "i": idx, "role": m.role, "name": m.name, "turn": m.turn,
                "tokens": count_message_tokens(m.content), "hash": content_hash(m.content),
                "call": bool(m.meta.get("tool_call")), "preview": _preview(m),
            })
            for f in self.facts:
                if f.id not in self._fact_source and m.role in ("user", "assistant") and f.found_in(m.content):
                    self._fact_source[f.id] = idx

    def step(self, user_text: str, plan: list[dict] | None = None, reply: str | None = None,
             new_session: bool = False) -> dict[str, Any]:
        if new_session:
            self.ctx.new_session()
            self.session += 1
        if isinstance(self.llm, ScriptedLLM):
            self.llm.set_plan(plan or [], reply)
        res = self.agent.chat(user_text)
        self._sync_messages()
        final = res.llm_calls[-1] if res.llm_calls else None
        by_hash = {}
        for m in self.messages:
            by_hash.setdefault((m["hash"], m["turn"]), m["i"])
        in_ctx: list[Any] = []
        system_tokens = 0
        for p in (final.prompt_parts if final else []):
            if p["role"] == "system" and p["turn"] == 0 and p.get("kind") is None:
                system_tokens = p["tokens"]
                continue
            idx = by_hash.get((p["hash"], p["turn"]))
            if idx is not None:
                in_ctx.append(idx)
            else:  # synthetic memory produced by the strategy (ledger, digests, redacted text)
                item = {"role": p["role"], "kind": p.get("kind") or "memory", "tokens": p["tokens"], "turn": p["turn"]}
                src = by_hash.get((p.get("src"), p["turn"])) if p.get("src") else None
                if src is None and p["role"] in ("user", "assistant"):
                    # a redacted copy of an original message: point at the original of that turn and role
                    src = next((m["i"] for m in self.messages if m["turn"] == p["turn"] and m["role"] == p["role"]
                                and not m["call"]), None)
                    item["kind"] = "redacted"
                if src is not None:
                    item["src"] = src
                in_ctx.append(item)
        decision = None
        evs = [e for e in getattr(self.ctx.primary, "events", []) if e.get("action") == "build"]
        if evs and evs[-1].get("turn") == res.turn:
            decision = evs[-1].get("summary")
        elif hasattr(self.ctx.primary, "dropped_messages") and self.ctx.primary.dropped_messages:
            decision = f"oldest {self.ctx.primary.dropped_messages} messages dropped to fit the window"
        prompt_text = final.prompt_text if final else ""
        facts = {}
        for f in self.facts:
            src = self._fact_source.get(f.id)
            if src is None:
                facts[f.id] = "in" if f.found_in(prompt_text) else "pending"
            else:
                facts[f.id] = "in" if f.found_in(prompt_text) else "lost"
        stale = {}
        for s in self.stale:
            if res.turn <= s.after_turn:
                stale[s.id] = "current"
            else:
                stale[s.id] = "leaked" if s.found_in(prompt_text) else "clean"
        tool_msgs = [m for m in self.messages if m["turn"] == res.turn and m["role"] == "tool"]
        record = {
            "t": res.turn,
            "user": user_text,
            "tools": [{"name": c.name, "args": c.args,
                       "tokens": tool_msgs[i]["tokens"] if i < len(tool_msgs) else 0}
                      for i, c in enumerate(res.tool_calls)],
            "reply": res.final_text,
            "prompt": final.prompt_tokens if final else 0,
            "system": system_tokens,
            "calls": len(res.llm_calls),
            "prompt_total": sum(c.prompt_tokens for c in res.llm_calls),
            "latency": round(sum(c.latency_s for c in res.llm_calls), 3),
            "raw": sum(m["tokens"] for m in self.messages),
            "ctx": in_ctx,
            "facts": facts,
            "stale": stale,
            "session": self.session,
            "decision": decision,
            "memory": self.ctx.primary.render_ledger() if hasattr(self.ctx.primary, "render_ledger") else None,
        }
        self.turns.append(record)
        return record

    def export(self) -> dict[str, Any]:
        memory = None
        if hasattr(self.ctx.primary, "render_ledger"):
            memory = self.ctx.primary.render_ledger()
        return {"label": self.label, "strategy": self.ctx.name, "max_ctx": self.ctx.max_context_tokens,
                "budget": self.ctx.prompt_budget, "messages": self.messages, "turns": self.turns,
                "fact_source": self._fact_source, "memory": memory}


def trace_scenario(scenario: Scenario, make_context: Callable[[], ContextManager],
                   backend: LLMBackend | None = None, max_steps: int = 5) -> dict[str, Any]:
    rec = TraceRecorder(make_context(), backend, scenario.retained, scenario.stale, max_steps)
    for turn in scenario.turns:
        rec.step(turn.user, turn.tools, turn.reply, new_session=turn.new_session)
    run = rec.export()
    last = rec.turns[-1] if rec.turns else {"facts": {}, "stale": {}, "reply": ""}
    retained_ok = all(v == "in" for v in last["facts"].values())
    leaked = any(v == "leaked" for v in last["stale"].values())
    answer = None
    if backend is not None:
        checks = [run_answer_check(c, last["reply"]) for c in scenario.answer]
        answer = {"passed": all(c["passed"] for c in checks), "checks": checks}
    run["summary"] = {
        "mode": "live" if backend is not None else "replay",
        "context_pass": retained_ok and not leaked,
        "task_pass": answer["passed"] if answer else (retained_ok and not leaked),
        "retention": (sum(v == "in" for v in last["facts"].values()) / len(last["facts"])) if last["facts"] else 1.0,
        "leaked": leaked,
        "total_prompt_tokens": sum(t["prompt_total"] for t in rec.turns),
        "peak_prompt": max((t["prompt"] for t in rec.turns), default=0),
        "answer": answer,
    }
    return run


def scenario_meta(s: Scenario) -> dict[str, Any]:
    return {"id": s.id, "title": s.title, "layer": s.layer, "description": s.description,
            "judge": s.judge, "turns": len(s.turns),
            "facts": [{"id": f.id, "label": f.display} for f in s.retained],
            "stale": [{"id": f.id, "label": f.display, "after_turn": f.after_turn} for f in s.stale]}
