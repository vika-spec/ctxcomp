"""Compact digests of tool outputs.

A 1,800-token hotel search becomes a ~150-token line that still names every
result with its price and rating, so the agent can refer back to it. The full
output stays in the archive.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..tokens import count_tokens

MAX_FALLBACK_CHARS = 400


def _price(it: dict[str, Any]) -> str:
    if it.get("price_per_night_usd") is not None:
        return f"${it['price_per_night_usd']}/night"
    if it.get("avg_price_usd") is not None:
        return f"~${it['avg_price_usd']}"
    if it.get("price_usd") is not None:
        return "free" if it["price_usd"] == 0 else f"${it['price_usd']}"
    return ""


def _places(d: dict[str, Any]) -> str:
    items = d.get("results") or []
    parts = []
    for it in items:
        bits = [b for b in (_price(it), f"{it['rating']}*" if it.get("rating") else "",
                            it.get("cuisine") or "", it.get("area") or "") if b]
        parts.append(f"{it.get('name')} ({', '.join(bits)})")
    return (f"places_search '{d.get('query')}' @ {d.get('location')}: {len(items)} {d.get('category', 'results')}: "
            + "; ".join(parts))


def _flights(d: dict[str, Any]) -> str:
    rs = d.get("results") or []
    top = [f"{r.get('airline')} ${r.get('price_usd')} ({r.get('stops')} stop{'s' if r.get('stops') != 1 else ''}, "
           f"{r.get('duration')}, dep {r.get('departure_local')})" for r in rs[:3]]
    route = rs[0].get("route") if rs else ""
    return f"flights {route}: {len(rs)} options; cheapest: " + "; ".join(top)


def _trains(d: dict[str, Any]) -> str:
    deps = d.get("departures") or []
    if not deps:
        return f"trains {d.get('route')}: no departures"
    fares = [x.get("fare_from_usd", 0) for x in deps]
    return (f"trains {d.get('route')}: {len(deps)} daily departures {deps[0]['departs']}-{deps[-1]['departs']}, "
            f"about {deps[0].get('duration')}, fares from ${min(fares)}; same timetable every day")


def _weather(d: dict[str, Any]) -> str:
    fc = d.get("forecast") or []
    if not fc:
        return f"weather {d.get('resolved_city')}: no data"
    highs = [f["high_c"] for f in fc]
    rainy = [f["day"].replace("Day ", "") for f in fc if "rain" in f.get("condition", "") or "shower" in f.get("condition", "")]
    return (f"weather {d.get('resolved_city')}: {len(fc)} days, highs {min(highs)}-{max(highs)}C, "
            f"rain on days {', '.join(rainy) if rainy else 'none'}")


def _web(d: dict[str, Any]) -> str:
    rs = d.get("results") or []
    return f"web search '{d.get('query')}': " + "; ".join(r.get("title", "") for r in rs[:3])


def digest(tool: str, raw: str) -> str:
    """One-line summary of a tool result. Short outputs are kept as they are."""
    if count_tokens(raw) <= 120:
        return re.sub(r"\n+", " | ", raw.strip())
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", raw)[:MAX_FALLBACK_CHARS] + " ..."
    if not isinstance(d, dict):
        return json.dumps(d)[:MAX_FALLBACK_CHARS]
    kind = d.get("type") or d.get("category")
    try:
        if tool == "places_search" or d.get("category") in ("hotels", "restaurants", "attractions"):
            return _places(d)
        if kind == "flight_results":
            return _flights(d)
        if kind == "rail_timetable":
            return _trains(d)
        if "forecast" in d:
            return _weather(d)
        if kind == "web_results":
            return _web(d)
    except (KeyError, TypeError):
        pass
    return re.sub(r"\s+", " ", json.dumps(d))[:MAX_FALLBACK_CHARS] + " ..."
