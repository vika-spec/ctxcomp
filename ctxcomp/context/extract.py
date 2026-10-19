"""Extract durable facts (constraints, preferences, commitments) from user messages.

Facts are stored as exact quotes from the user's own words, never paraphrases,
so compression cannot distort them. Two extractors:

- HeuristicExtractor: sentence classifier built on general patterns (allergies,
  diets, budget caps, fixed appointments, pace, mobility, party, itinerary,
  preferences). Fast, deterministic, no model needed.
- LLMExtractor: asks the agent's own model for constraints, but only accepts
  quotes that appear verbatim in the message (a faithfulness guard). It is
  combined with the heuristic extractor, never used alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DAY = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|weekend)"
TIME = r"(\d{1,2}(:\d{2})?\s?(am|pm)|noon|midnight)"
MONEY = r"[$€£₹]\s?\d[\d,]*(\.\d+)?\s?k?"

# Ordered by priority: a sentence gets every kind it matches; the first is its primary kind.
KIND_PATTERNS: list[tuple[str, re.Pattern]] = [(k, re.compile(p, re.I)) for k, p in [
    ("allergy", r"\ballerg|\bintoleran|\bceliac\b|\bcoeliac\b|anaphyla|epipen"),
    ("diet", r"\bvegan\b|\bvegetarian\b|\bhalal\b|\bkosher\b|\bpescatarian\b|\bjain\b|gluten|lactose|dairy-free|"
             r"\b(can'?t|cannot|don'?t|do not|won'?t|never) eat\b|\bno (pork|beef|meat|alcohol|nuts|dairy|seafood|fish)\b"),
    ("budget", rf"\b(budget|spending cap|cap of|max(imum)?|at most|no more than|up to|afford|spend|under|below|"
               rf"less than)\b[^.!?\n]{{0,30}}{MONEY}|"
               rf"{MONEY}\s+(budget|max|maximum|cap)\b"),
    ("appointment", rf"\b(meeting|appointment|conference|wedding|interview|call|reservation|ceremony|concert|"
                    rf"event|presentation|class|exam)\b[^.!?\n]{{0,80}}\b({DAY}|{TIME})"),
    ("mobility", r"wheelchair|mobility|\bstairs\b|can'?t walk|cannot walk|limited walking|\bknee\b|\bstroller\b|"
                 r"\belderly\b|\baccessib|\bcane\b|\bcrutches\b"),
    ("pace", r"\brelax|\bpacked\b|\bpace\b|per day\b|\bhectic\b|\brushed\b|\bdowntime\b|\bearly (mornings?|starts?)\b|"
             r"\bslow (travel|trip|pace)\b"),
    ("party", r"\bsolo\b|\btravell?ing with\b|\bwith my (wife|husband|partner|kids|children|family|parents|friends?|"
              r"son|daughter|mom|dad)\b|\b\d+ (people|adults|travell?ers|kids|children)\b|\bfamily of\b"),
    ("itinerary", rf"\b\d+[- ]?(days?|nights?)\b|\b\d+ in (?-i:[A-Z])|"
                  rf"\b(arriv(e|es|ing)|land(s|ing)?|get(ting)? (in|into))\b[^.!?\n]{{0,40}}\b({DAY}|\d)|"
                  rf"\bdepart(ing|ure)?\b[^.!?\n]{{0,40}}\b({DAY}|\d)|\bleaving on\b"),
    ("preference", r"\bi (really )?(want|prefer|love|like|hate|need)\b|\bi'?d (like|love|prefer|rather)\b|"
                   r"\bnot a fan\b|\bno (beaches|crowds|museums|hostels|nightlife|tours)\b|, not \w+"),
]]

# Request sentences ("Find hotels...", "Book the...") and questions only count
# if they carry a hard constraint.
REQUEST_RE = re.compile(r"^\s*(please\s+)?(find|show|book|search|look|get|give|list|what|where|when|how|which|who|"
                        r"any|is|are|can|could|would|do|does|tell|check|recommend|suggest|ok|okay|and|now)\b", re.I)
HARD_KINDS = {"allergy", "diet", "budget", "price_limit", "appointment", "mobility"}

PIVOT_RES = [re.compile(p) for p in [
    r"\b(?i:scratch|cancel|forget(?: about)?|scrap|ditch|drop|skip|abandon)\s+(?:the\s+)?(?:whole\s+)?"
    r"(?P<x>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})",
    r"\b(?i:instead of|rather than)\s+(?P<x>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})",
    r"\b(?i:no longer|not)\s+(?i:going to|doing|visiting)\s+(?P<x>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})",
]]
PIVOT_STOP = {"I", "The", "That", "This", "It", "All", "Everything", "My", "Our"}


@dataclass
class Fact:
    text: str                    # exact quote from the user
    kinds: list[str]
    turn: int
    active: bool = True
    note: str | None = None      # e.g. "updated at turn 3"
    source: str = "heuristic"

    @property
    def kind(self) -> str:
        return self.kinds[0]

    def to_dict(self) -> dict[str, Any]:
        return dict(text=self.text, kinds=self.kinds, turn=self.turn, active=self.active, note=self.note,
                    source=self.source)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        return cls(**d)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text or "") if s.strip()]


PER_ITEM_RE = re.compile(r"\bper (night|day|person|meal|ticket)\b|\ba night\b|\beach\b|/night|\bnightly\b", re.I)
CAP_WORD_RE = re.compile(r"\b(budget|cap|maximum|max)\b", re.I)


def classify(sentence: str) -> list[str]:
    kinds = [k for k, rx in KIND_PATTERNS if rx.search(sentence)]
    if not kinds:
        return []
    is_question = sentence.rstrip().endswith("?")
    is_request = bool(REQUEST_RE.match(sentence)) or is_question
    if "budget" in kinds:
        if PER_ITEM_RE.search(sentence):
            # "under $200 a night" limits one item; it must not replace the trip budget
            kinds = ["price_limit" if k == "budget" else k for k in kinds]
        elif is_question or (is_request and not CAP_WORD_RE.search(sentence)):
            # "Can we afford a $250 dinner?" asks about the budget; it does not set one
            kinds = [k for k in kinds if k != "budget"]
    if is_request:
        kinds = [k for k in kinds if k in HARD_KINDS]
    return kinds


def find_pivots(text: str) -> list[str]:
    """Names of plans the user just cancelled, e.g. 'Bali' in 'scratch Bali entirely'."""
    out = []
    for rx in PIVOT_RES:
        for m in rx.finditer(text or ""):
            x = m.group("x").strip()
            words = x.split()
            while words and words[-1].lower() in {"entirely", "completely", "altogether", "instead"}:
                words.pop()
            x = " ".join(words)
            if x and x not in PIVOT_STOP and x not in out:
                out.append(x)
    return out


def parse_money(text: str) -> float | None:
    m = re.search(r"[$€£₹]\s?(\d[\d,]*(?:\.\d+)?)\s?(k)?\b", text or "", re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * 1000 if m.group(2) else v


class HeuristicExtractor:
    name = "heuristic"

    def __call__(self, text: str, turn: int) -> list[Fact]:
        facts = []
        for s in split_sentences(text):
            if find_pivots(s):
                continue  # a cancellation is handled as a pivot, not stored as a fact
            kinds = classify(s)
            if kinds:
                facts.append(Fact(text=s, kinds=kinds, turn=turn))
        return facts


class LLMExtractor:
    """Model-assisted extraction with a verbatim-quote guard. Falls back to the
    heuristic extractor's output if the model returns nothing usable."""

    name = "llm"
    PROMPT = (
        "Extract every durable constraint or preference the user states in the message below: allergies, "
        "diets, budgets, fixed appointments, pace, mobility, who is travelling, trip length and dates, likes "
        "and dislikes. Ignore requests and questions. Reply with ONLY a JSON list like "
        '[{"kind": "allergy", "quote": "<exact words copied from the message>"}]. '
        "Copy quotes exactly. Reply [] if there are none.\n\nMessage: "
    )

    def __init__(self, backend) -> None:
        self.backend = backend
        self.heuristic = HeuristicExtractor()

    def __call__(self, text: str, turn: int) -> list[Fact]:
        from ..messages import Message
        base = self.heuristic(text, turn)
        try:
            out = self.backend.generate([Message("user", self.PROMPT + text, turn)], step=-1).text
            start, end = out.find("["), out.rfind("]")
            items = json.loads(out[start:end + 1]) if start >= 0 and end > start else []
        except Exception:
            return base
        have = {f.text for f in base}
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            quote = str(it.get("quote", "")).strip()
            kind = str(it.get("kind", "preference")).lower().strip() or "preference"
            if len(quote) < 4 or quote not in text:  # faithfulness guard: verbatim only
                continue
            if any(quote in h or h in quote for h in have):
                continue
            base.append(Fact(text=quote, kinds=[kind], turn=turn, source="llm"))
            have.add(quote)
        return base
