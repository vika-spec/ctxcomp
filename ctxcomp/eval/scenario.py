"""Scenario loading and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "scenarios"


@dataclass
class Turn:
    user: str
    tools: list[dict[str, Any]] = field(default_factory=list)   # scripted calls (replay mode only)
    reply: str | None = None                                     # scripted reply (replay mode only)
    new_session: bool = False                                    # user returns later: context starts a new session


@dataclass
class FactCheck:
    id: str
    any: list[str]
    label: str | None = None
    after_turn: int = 0          # stale checks: the info only becomes stale after this turn

    def found_in(self, text: str) -> bool:
        return any(re.search(p, text, re.I) for p in self.any)

    @property
    def display(self) -> str:
        return self.label or self.id.replace("_", " ").capitalize()


def _fact(c: dict) -> FactCheck:
    return FactCheck(c["id"], c["any"], c.get("label"), int(c.get("after_turn", 0)))


@dataclass
class Scenario:
    id: str
    title: str
    layer: str
    description: str
    turns: list[Turn]
    retained: list[FactCheck]
    stale: list[FactCheck]
    answer: list[dict[str, Any]]
    judge: str | None = None
    path: str | None = None


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    for key in ("id", "turns", "checks"):
        if key not in raw:
            raise ValueError(f"{path.name}: missing '{key}'")
    turns = [Turn(user=t["user"], tools=t.get("tools") or [], reply=t.get("reply"),
                  new_session=bool(t.get("new_session"))) for t in raw["turns"]]
    checks = raw["checks"] or {}
    sc = Scenario(
        id=raw["id"], title=raw.get("title", raw["id"]), layer=str(raw.get("layer", "")),
        description=" ".join(str(raw.get("description", "")).split()),
        turns=turns,
        retained=[_fact(c) for c in checks.get("retained") or []],
        stale=[_fact(c) for c in checks.get("stale") or []],
        answer=list(checks.get("answer") or []),
        judge=" ".join(str(raw["judge"]).split()) if raw.get("judge") else None,
        path=str(path),
    )
    for pattern_list in [c.any for c in sc.retained + sc.stale]:
        for p in pattern_list:
            re.compile(p)
    for chk in sc.answer:
        if chk.get("type") not in ("forbid", "forbid_unless", "require_any"):
            raise ValueError(f"{path.name}: unknown answer check type {chk.get('type')!r}")
    return sc


def load_scenarios(directory: str | Path = DEFAULT_DIR, only: list[str] | None = None) -> list[Scenario]:
    paths = sorted(Path(directory).glob("*.yaml"))
    scenarios = [load_scenario(p) for p in paths]
    if only and only != ["all"]:
        wanted = {o.lower() for o in only}
        scenarios = [s for s in scenarios
                     if s.id.lower() in wanted or s.id.split("_")[0].lower() in wanted]
    return scenarios
