"""CompressedContext: tiered memory for tool-using agents under a small context window.

How the prompt is assembled, every model call, within a fixed token budget:

  1. System prompt.
  2. Pinned memory (the ledger), never evicted:
       - user constraints and preferences, stored as exact quotes
       - numeric budget state: cap, spent so far, remaining (computed, not summarized)
       - bookings, and a note when an earlier plan was cancelled
  3. The current turn, verbatim (user message, tool calls, full tool results).
  4. Relevant older turns, retrieved by keyword relevance to the current request.
  5. Recent older turns, newest first, until the budget is used up.

Older turns are compacted: user messages and replies are kept, tool results are
replaced by one-line digests (identical results are deduplicated), and the raw
outputs stay in the archive. Turns about a cancelled plan are invalidated and
never shown again, and superseded values (an old budget) are redacted from the
history. Every build is logged so the compression decisions are inspectable.

Ablation switches (use_ledger, use_digests, use_invalidation, use_retrieval)
turn components off to measure what each one contributes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..messages import Message, ToolCall, content_hash
from ..tokens import count_message_tokens
from .base import ContextManager
from .digest import digest
from .extract import Fact, HeuristicExtractor, find_pivots, parse_money, split_sentences

HARD_ORDER = ["allergy", "diet", "budget", "price_limit", "appointment", "mobility"]
SOFT_ORDER = ["party", "pace", "itinerary", "preference"]
KIND_RANK = {k: i for i, k in enumerate(HARD_ORDER + SOFT_ORDER)}
PLAN_KINDS = {"itinerary", "preference", "pace"}   # plan-specific; invalidated with a cancelled plan
STOPWORDS = set("""the and for with from that this what where when which about there their them they you your
are was were can could would should will just any some find show book get give tell me my our its also into
near best good great more most very much than then trip plan""".split())


@dataclass
class ToolEntry:
    name: str
    args: dict[str, Any]
    raw: str | None = None
    digest: str = ""
    hash: str = ""
    dup_of: int | None = None
    raw_tokens: int = 0


@dataclass
class TurnRec:
    turn: int
    session: int
    user: str
    calls: list[ToolEntry] = field(default_factory=list)
    reply: str | None = None


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in STOPWORDS]


class CompressedContext(ContextManager):
    name = "compressed"
    use_ledger = True
    use_digests = True
    use_invalidation = True
    use_retrieval = True

    def __init__(self, max_context_tokens: int = 8192, reserve_output_tokens: int = 512,
                 extractor=None, ledger_share: float = 0.25, retrieval_share: float = 0.3,
                 retrieval_k: int = 3) -> None:
        super().__init__(max_context_tokens, reserve_output_tokens)
        self.extractor = extractor or HeuristicExtractor()
        self.ledger_share, self.retrieval_share, self.retrieval_k = ledger_share, retrieval_share, retrieval_k
        self.turns: dict[int, TurnRec] = {}
        self.facts: list[Fact] = []
        self.cancelled: dict[str, int] = {}      # cancelled plan name -> turn it was cancelled
        self.stale_turns: set[int] = set()
        self.redactions: dict[str, str] = {}     # superseded sentence -> replacement
        self.bookings: list[dict[str, Any]] = []
        self.spent: float | None = None
        self.seen: dict[str, int] = {}           # tool output hash -> first turn it appeared
        self.current_turn = 0
        self.session = 1
        self.raw_tokens = 0
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ events
    def on_user(self, text: str, turn: int) -> None:
        self.current_turn = turn
        self.turns[turn] = TurnRec(turn, self.session, text)
        self.raw_tokens += count_message_tokens(text)

        if self.use_invalidation:
            # A cancelled plan mentioned again without a cancellation is back on.
            for name in list(self.cancelled):
                if re.search(rf"\b{re.escape(name)}\b", text, re.I) and name not in find_pivots(text):
                    del self.cancelled[name]
            for name in find_pivots(text):
                self.cancelled[name] = turn
                self._invalidate(name, turn)

        for f in self.extractor(text, turn):
            if any(f.text == g.text for g in self.facts):
                continue
            if self.use_invalidation and "budget" in f.kinds:
                self._supersede_budget(f)
            self.facts.append(f)

    def on_tool_call(self, call: ToolCall, turn: int) -> None:
        rec = self.turns.get(turn) or self.turns.setdefault(turn, TurnRec(turn, self.session, ""))
        rec.calls.append(ToolEntry(call.name, dict(call.args)))
        self.raw_tokens += count_message_tokens(call.to_text())

    def on_tool_result(self, call: ToolCall, result: str, turn: int) -> None:
        rec = self.turns[turn]
        entry = next((e for e in reversed(rec.calls) if e.raw is None and e.name == call.name), None)
        if entry is None:
            entry = ToolEntry(call.name, dict(call.args))
            rec.calls.append(entry)
        entry.raw, entry.hash = result, content_hash(result)
        entry.raw_tokens = count_message_tokens(result)
        entry.digest = digest(call.name, result)
        key = self._dedup_key(result)
        first = self.seen.get(key)
        if first is not None and first != turn and first not in self.stale_turns:
            entry.dup_of = first
        else:
            self.seen[key] = turn
        self.raw_tokens += entry.raw_tokens
        if call.name == "budget_tracker":
            self._track_budget(call, result, turn)

    def on_assistant(self, text: str, turn: int) -> None:
        rec = self.turns.get(turn)
        if rec is not None:
            rec.reply = text
        self.raw_tokens += count_message_tokens(text)

    @staticmethod
    def _dedup_key(result: str) -> str:
        """Results listing the same set of places count as duplicates, even if the
        query wording differed (e.g. 'top attractions' vs 'things to do')."""
        try:
            d = json.loads(result)
            names = sorted(str(r.get("name")) for r in d.get("results", []) if isinstance(r, dict) and r.get("name"))
            if len(names) >= 3:
                return "names:" + content_hash("|".join(names))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return content_hash(result)

    # --------------------------------------------------------- state updates
    def _invalidate(self, name: str, pivot_turn: int) -> None:
        """Mark earlier turns about a cancelled plan as stale."""
        key = name.lower()
        for t, rec in self.turns.items():
            if t >= pivot_turn:
                continue
            blob = (rec.user + " " + " ".join(json.dumps(e.args) for e in rec.calls)).lower()
            if key in blob:
                self.stale_turns.add(t)
        for f in self.facts:
            if f.turn in self.stale_turns and f.kind in PLAN_KINDS:
                f.active, f.note = False, f"cancelled at turn {pivot_turn}"
        self.log(action="invalidate", turn=pivot_turn, cancelled=name, stale_turns=sorted(self.stale_turns))

    def _supersede_budget(self, new: Fact) -> None:
        money = re.compile(r"[$€£₹]\s?\d[\d,]*(\.\d+)?\s?k?\b")
        for old in self.facts:
            if not old.active or "budget" not in old.kinds:
                continue
            redacted = money.sub("[amount updated later]", old.text)
            self.redactions[old.text] = redacted
            # The old amount may also be echoed elsewhere (e.g. in the agent's reply).
            for m in money.finditer(old.text):
                if parse_money(m.group(0)) != parse_money(new.text):
                    self.redactions[m.group(0).strip()] = "[amount updated later]"
            others = [k for k in old.kinds if k != "budget"]
            if others:  # keep the rest of the sentence ("I'm a solo traveler")
                old.text, old.kinds = redacted, others
            else:
                old.active = False
            old.note = f"budget updated at turn {new.turn}"

    def _track_budget(self, call: ToolCall, result: str, turn: int) -> None:
        m = re.search(r"TOTAL SPENT:\s*[$€£₹]?([\d,]+(?:\.\d+)?)", result)
        if m:
            self.spent = float(m.group(1).replace(",", ""))
        action = str(call.args.get("action", "")).lower()
        if action in ("add", "add_expense") and call.args.get("amount") is not None:
            self.bookings.append({"item": str(call.args.get("item")), "amount": float(call.args["amount"]), "turn": turn})
        elif action in ("remove", "remove_expense"):
            self.bookings = [b for b in self.bookings if b["item"].lower() != str(call.args.get("item", "")).lower()]
        elif action == "reset":
            self.bookings = []

    # -------------------------------------------------------------- rendering
    def _redact(self, text: str) -> str:
        for old, new in self.redactions.items():
            text = text.replace(old, new)
        for name in self.cancelled:
            sentences = split_sentences(text)
            if any(re.search(rf"\b{re.escape(name)}\b", s, re.I) for s in sentences):
                text = " ".join("[mention of a cancelled plan removed]"
                                if re.search(rf"\b{re.escape(name)}\b", s, re.I) else s for s in sentences)
        return text

    def render_ledger(self, max_tokens: int | None = None) -> str:
        # Hard constraints survive a cancelled plan (an allergy still applies); plan details do not.
        facts = sorted((f for f in self.facts if f.active),
                       key=lambda f: (KIND_RANK.get(f.kind, 99), f.turn))
        cap = None
        for f in self.facts:
            if f.active and "budget" in f.kinds:
                cap = parse_money(f.text) or cap
        bookings = [b for b in self.bookings if b["turn"] not in self.stale_turns]

        def build(fs: list[Fact]) -> str:
            lines = ["MEMORY (maintained across the conversation; quotes are the user's own words, follow them):"]
            hard = [f for f in fs if f.kind in HARD_ORDER]
            soft = [f for f in fs if f.kind not in HARD_ORDER]
            if hard:
                lines.append("Constraints:")
                lines += [f'- turn {f.turn}: "{f.text}"' for f in hard]
            if soft:
                lines.append("Trip details and preferences:")
                lines += [f'- turn {f.turn}: "{f.text}"' for f in soft]
            if cap is not None or self.spent is not None:
                spent = self.spent or 0.0
                line = f"Budget: total spent ${spent:,.0f}"
                if cap is not None:
                    left = cap - spent
                    remaining = f"remaining ${left:,.0f}" if left >= 0 else f"OVER BUDGET by ${-left:,.0f}"
                    line = f"Budget: cap ${cap:,.0f} | total spent ${spent:,.0f} | {remaining}"
                lines.append(line)
            if bookings:
                lines.append("Booked so far: " + "; ".join(f"{b['item']} ${b['amount']:,.0f} (turn {b['turn']})"
                                                           for b in bookings))
            if self.cancelled:
                turns = sorted(set(self.cancelled.values()))
                lines.append(f"Plan change: at turn {', '.join(map(str, turns))} the user cancelled an earlier plan. "
                             "Do not mention or reuse anything from the cancelled plan.")
            if self.session > 1:
                lines.append(f"This is session {self.session}. Earlier sessions are summarized in this memory "
                             "and in the compact history below.")
            return "\n".join(lines) if len(lines) > 1 else ""

        text = build(facts)
        if max_tokens is not None:
            # Over budget: drop the oldest soft facts first; hard constraints are never dropped.
            soft = [f for f in facts if f.kind not in HARD_ORDER]
            while soft and count_message_tokens(text) > max_tokens:
                facts.remove(soft.pop(0))
                text = build(facts)
        return text

    def _turn_messages(self, t: int, current: bool) -> list[Message]:
        rec = self.turns[t]
        msgs: list[Message] = []
        if rec.user:
            msgs.append(Message("user", rec.user if current else self._redact(rec.user), t))
        for e in rec.calls:
            if current:
                msgs.append(Message("assistant", ToolCall(e.name, e.args).to_text(), t, meta={"tool_call": True}))
                if e.raw is not None:
                    msgs.append(Message("tool", e.raw, t, name=e.name))
                continue
            if not self.use_digests and e.raw is not None:
                msgs.append(Message("tool", e.raw, t, name=e.name))
                continue
            if e.dup_of is not None:
                text = f"(same places as the turn {e.dup_of} {e.name} results)"
            else:
                text = f"(digest of turn {t} result) {e.digest}"
            msgs.append(Message("tool", text, t, name=e.name, meta={"kind": "digest", "src_hash": e.hash}))
        if rec.reply and not current:
            msgs.append(Message("assistant", self._redact(rec.reply), t))
        return msgs

    def _relevance(self, query: str, candidates: dict[int, list[Message]]) -> list[tuple[float, int]]:
        q = set(_words(query))
        if not q or not candidates:
            return []
        docs = {t: set(_words(" ".join(m.content for m in ms))) for t, ms in candidates.items()}
        n = len(docs)
        df = {w: sum(1 for d in docs.values() if w in d) for w in q}
        scored = []
        for t, d in docs.items():
            s = sum(math.log(1 + n / df[w]) for w in q if w in d and df[w])
            if s > 0:
                scored.append((s, t))
        return sorted(scored, reverse=True)

    def build(self, system_prompt: str) -> list[Message]:
        budget = self.prompt_budget
        system = Message("system", system_prompt, 0)
        used = count_message_tokens(system_prompt)
        out: list[Message] = [system]

        ledger_tokens = 0
        if self.use_ledger:
            text = self.render_ledger(max_tokens=int(budget * self.ledger_share))
            if text:
                out.append(Message("system", text, 0, meta={"kind": "ledger"}))
                ledger_tokens = count_message_tokens(text)
                used += ledger_tokens

        cur = self.current_turn
        current = self._turn_messages(cur, current=True) if cur in self.turns else []
        cur_tokens = sum(count_message_tokens(m.content) for m in current)
        if used + cur_tokens > budget:
            current = self._shrink_current(current, budget - used)
            cur_tokens = sum(count_message_tokens(m.content) for m in current)
        avail = budget - used - cur_tokens

        older = [t for t in sorted(self.turns) if t < cur
                 and not (self.use_invalidation and t in self.stale_turns)]
        bundles = {t: self._turn_messages(t, current=False) for t in older}
        size = {t: sum(count_message_tokens(m.content) for m in ms) for t, ms in bundles.items()}
        chosen: dict[int, list[Message]] = {}
        retrieved: list[int] = []

        if self.use_retrieval and cur in self.turns and avail > 0:
            cap = int(avail * self.retrieval_share)
            spent = 0
            for score, t in self._relevance(self.turns[cur].user, bundles)[: self.retrieval_k]:
                if spent + size[t] <= cap:
                    chosen[t] = bundles[t]
                    spent += size[t]
                    retrieved.append(t)
            avail -= spent

        for t in reversed(older):
            if t in chosen:
                continue
            if size[t] <= avail:
                chosen[t] = bundles[t]
                avail -= size[t]
                continue
            slim = [m for m in bundles[t] if m.role != "tool"]   # user + reply only
            slim_size = sum(count_message_tokens(m.content) for m in slim)
            if slim and slim_size <= avail:
                chosen[t] = slim
                avail -= slim_size
                continue
            break

        for t in sorted(chosen):
            out.extend(chosen[t])
        out.extend(current)

        dropped = [t for t in older if t not in chosen]
        prompt_tokens = sum(count_message_tokens(m.content) for m in out)
        self.last = {
            "turn": cur, "ledger_tokens": ledger_tokens, "kept_turns": sorted(chosen), "retrieved": retrieved,
            "dropped_turns": dropped, "stale_turns": sorted(self.stale_turns), "prompt_tokens": prompt_tokens,
            "digests": sum(1 for t in chosen for m in chosen[t] if m.meta.get("kind") == "digest"),
        }
        self.log(action="build", summary=self.explain(), **self.last)
        return out

    def _shrink_current(self, current: list[Message], room: int) -> list[Message]:
        """The current turn alone overflows: digest its earlier tool results, then trim the newest."""
        tools = [i for i, m in enumerate(current) if m.role == "tool"]
        for i in tools[:-1]:
            m = current[i]
            current[i] = Message("tool", digest(m.name or "", m.content), m.turn, m.name,
                                 meta={"kind": "digest", "src_hash": content_hash(m.content)})
            if sum(count_message_tokens(x.content) for x in current) <= room:
                return current
        if tools:
            i = tools[-1]
            over = sum(count_message_tokens(x.content) for x in current) - room
            m = current[i]
            keep = max(200, len(m.content) - over * 4 - 40)
            current[i] = Message("tool", m.content[:keep] + "\n...(truncated to fit)", m.turn, m.name, dict(m.meta))
        return current

    def explain(self) -> str:
        L = self.last
        if not L:
            return ""
        parts = [f"pinned memory {L['ledger_tokens']} tok" if L["ledger_tokens"] else "no pinned memory"]
        parts.append(f"{len(L['kept_turns'])} earlier turns kept ({L['digests']} tool results as digests)")
        if L["retrieved"]:
            parts.append("retrieved turns " + ", ".join(map(str, L["retrieved"])))
        if L["dropped_turns"]:
            parts.append(f"{len(L['dropped_turns'])} older turns left out")
        if L["stale_turns"]:
            parts.append("cancelled-plan turns ignored: " + ", ".join(map(str, L["stale_turns"])))
        return "; ".join(parts)

    def stats(self) -> dict[str, Any]:
        return {
            "history_tokens": self.raw_tokens,
            "active_facts": sum(1 for f in self.facts if f.active),
            "cancelled": dict(self.cancelled),
            "compression_ratio": round(self.raw_tokens / self.last["prompt_tokens"], 2)
            if self.last.get("prompt_tokens") else None,
            **{k: v for k, v in self.last.items() if k != "turn"},
        }

    # ------------------------------------------------------------- sessions
    def to_state(self) -> dict[str, Any]:
        """Everything needed to continue in a new session (raw tool outputs are not kept)."""
        return {
            "version": 1, "session": self.session, "current_turn": self.current_turn,
            "facts": [f.to_dict() for f in self.facts], "cancelled": self.cancelled,
            "stale_turns": sorted(self.stale_turns), "redactions": self.redactions,
            "bookings": self.bookings, "spent": self.spent, "seen": self.seen, "raw_tokens": self.raw_tokens,
            "turns": [{"turn": r.turn, "session": r.session, "user": r.user, "reply": r.reply,
                       "calls": [{"name": e.name, "args": e.args, "digest": e.digest, "hash": e.hash,
                                  "dup_of": e.dup_of} for e in r.calls]} for r in self.turns.values()],
        }

    def load_state(self, state: dict[str, Any], new_session: bool = True) -> None:
        self.session = int(state.get("session", 1)) + (1 if new_session else 0)
        self.current_turn = int(state.get("current_turn", 0))
        self.facts = [Fact.from_dict(d) for d in state.get("facts", [])]
        self.cancelled = dict(state.get("cancelled", {}))
        self.stale_turns = set(state.get("stale_turns", []))
        self.redactions = dict(state.get("redactions", {}))
        self.bookings = list(state.get("bookings", []))
        self.spent = state.get("spent")
        self.seen = dict(state.get("seen", {}))
        self.raw_tokens = int(state.get("raw_tokens", 0))
        self.turns = {}
        for r in state.get("turns", []):
            rec = TurnRec(r["turn"], r["session"], r["user"], reply=r.get("reply"))
            rec.calls = [ToolEntry(c["name"], c["args"], None, c["digest"], c["hash"], c.get("dup_of")) for c in r["calls"]]
            self.turns[rec.turn] = rec

    def new_session(self) -> "CompressedContext":
        state = json.loads(json.dumps(self.to_state()))   # proves the state is serializable
        fresh = type(self)(self.max_context_tokens, self.reserve_output_tokens, extractor=self.extractor,
                           ledger_share=self.ledger_share, retrieval_share=self.retrieval_share,
                           retrieval_k=self.retrieval_k)
        fresh.load_state(state, new_session=True)
        fresh.log(action="new_session", session=fresh.session, restored_turns=len(fresh.turns))
        return fresh


# Ablations: the same module with one component switched off.
class NoLedger(CompressedContext):
    name = "compressed-no-ledger"
    use_ledger = False


class NoDigests(CompressedContext):
    name = "compressed-no-digests"
    use_digests = False


class NoInvalidation(CompressedContext):
    name = "compressed-no-invalidation"
    use_invalidation = False


class NoRetrieval(CompressedContext):
    name = "compressed-no-retrieval"
    use_retrieval = False


ABLATIONS = [NoLedger, NoDigests, NoInvalidation, NoRetrieval]
