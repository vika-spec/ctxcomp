"""Rule-based travel tools.

The tools are deterministic and return realistic, deliberately verbose payloads,
because verbose tool output is exactly what fills a small context window in
real agents. Each tool returns a string (JSON text), like a real API would.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import data as D


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., str]

    def render(self) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({params}): {self.description}"


@dataclass
class ToolRegistry:
    """Holds tool specs plus any per-conversation state (the budget ledger)."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def render(self) -> str:
        return "\n".join(t.render() for t in self.tools.values())

    def call(self, name: str, args: dict[str, Any]) -> str:
        spec = self.tools.get(name)
        if spec is None:
            return json.dumps({"error": f"unknown tool '{name}'", "available": list(self.tools)})
        try:
            return spec.fn(**(args or {}))
        except TypeError as e:
            return json.dumps({"error": f"bad arguments for {name}: {e}"})
        except Exception as e:  # tools must never crash the agent loop
            return json.dumps({"error": f"{name} failed: {e}"})


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# web_search: flights, trains and general information
# ---------------------------------------------------------------------------

AIRLINES = ["Pacific Crest Air", "Meridian Airways", "Northstar Airlines", "Aurora Air",
            "Skyline International", "Blue Harbor Airlines", "Transalpine Air"]


def _parse_route(query: str) -> tuple[str, str]:
    q = query.lower()
    m = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:on|in|for|next|this|around)\b|$)", q)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"(.+?)\s+to\s+(.+?)(?:\s+(?:on|in|for|next|this|around)\b|$)", q)
    if m:
        return m.group(1).replace("flights", "").replace("flight", "").replace("train", "").strip(), m.group(2)
    return "san francisco", q.replace("flights", "").replace("flight", "").strip() or "unknown"


def _flights(query: str) -> str:
    o_raw, d_raw = _parse_route(query)
    origin, _ = D.resolve_city(o_raw)
    dest, _ = D.resolve_city(d_raw)
    r = D.rng_for("flight", origin, dest)
    o_code = D.AIRPORTS.get(origin, origin[:3].upper())
    d_code = D.AIRPORTS.get(dest, dest[:3].upper())
    long_haul = D.country_of(origin) != D.country_of(dest) and "usa" in (D.country_of(origin), D.country_of(dest))
    base = r.randint(620, 980) if long_haul else r.randint(70, 260)
    results = []
    for i in range(6):
        stops = r.choice([0, 0, 1, 1, 2]) if long_haul else r.choice([0, 0, 1])
        price = base + r.randint(-60, 380) + (0 if stops else 140)
        dep_h = r.randint(6, 22)
        dur = (r.randint(10, 14) if long_haul else r.randint(1, 3)) + stops * r.randint(1, 3)
        layover = None
        if stops:
            hubs = ["Frankfurt (FRA)", "Seoul (ICN)", "Dubai (DXB)", "London (LHR)", "Doha (DOH)", "Munich (MUC)"]
            layover = [f"{r.choice(hubs)}, {r.randint(1, 4)}h {r.choice([5, 20, 35, 50])}m" for _ in range(stops)]
        results.append({
            "option": i + 1,
            "airline": r.choice(AIRLINES),
            "flight_numbers": [f"{r.choice('PMNASBT')}{r.choice('CAXHTL')}{r.randint(100, 999)}" for _ in range(stops + 1)],
            "route": f"{o_code} -> {d_code}",
            "departure_local": f"{dep_h:02d}:{r.choice(['05', '20', '35', '50'])}",
            "duration": f"{dur}h {r.choice([0, 15, 30, 45])}m",
            "stops": stops,
            "layovers": layover,
            "cabin": "economy",
            "price_usd": price,
            "baggage": r.choice([
                "1 personal item included; carry-on 7kg included; first checked bag 23kg costs extra.",
                "1 carry-on and 1 checked bag (23kg) included in fare.",
                "Personal item only; carry-on and checked bags are paid add-ons at booking.",
            ]),
            "fare_rules": r.choice([
                "Changes allowed with fee plus fare difference. Non-refundable. Seat selection from check-in.",
                "Flexible fare: free changes up to 24h before departure, refundable as travel credit.",
                "Basic fare: no changes, no refunds, boarding in final group, no advance seat selection.",
            ]),
            "on_time_performance": f"{r.randint(68, 94)}%",
            "aircraft": r.choice(["Airbus A350-900", "Boeing 787-9", "Airbus A321neo", "Boeing 777-300ER", "Airbus A220"]),
        })
    results.sort(key=lambda x: x["price_usd"])
    return _dump({
        "query": query,
        "type": "flight_results",
        "currency": "USD",
        "results": results,
        "notes": [
            "Prices are per adult, one way, including taxes and carrier surcharges.",
            "Fares change frequently; book within 24 hours to hold this price.",
            "Some itineraries combine separate tickets; missed connections may not be protected.",
        ],
    })


def _trains(query: str) -> str:
    o_raw, d_raw = _parse_route(query)
    origin, _ = D.resolve_city(o_raw)
    dest, _ = D.resolve_city(d_raw)
    r = D.rng_for("train", origin, dest)
    hours = r.randint(1, 4)
    departures = []
    for h in range(6, 21, r.choice([1, 2])):
        departures.append({
            "departs": f"{h:02d}:{r.choice(['07', '13', '25', '43', '55'])}",
            "arrives": f"{(h + hours) % 24:02d}:{r.choice(['02', '18', '31', '46'])}",
            "duration": f"{hours}h {r.randint(5, 55)}m",
            "changes": r.choice([0, 0, 0, 1]),
            "fare_from_usd": r.randint(35, 70) + (0 if h < 9 or h > 18 else r.randint(20, 90)),
        })
    return _dump({
        "query": query,
        "type": "rail_timetable",
        "route": f"{origin.title()} -> {dest.title()}",
        "operator": "High-speed rail (sample data)",
        "frequency": "Departures run every day of the week on the same timetable.",
        "departures": departures,
        "fare_classes": {
            "standard": "Reserved seat, 1 large bag + 1 small bag, onboard cafe car.",
            "comfort": "Wider seats, at-seat power, light meal on morning services.",
            "premier": "Lounge access at both stations, full meal, flexible changes.",
        },
        "tips": [
            "Early morning and late evening departures are usually the cheapest.",
            "Arrive at least 30 minutes early; stations run security and passport checks on some routes.",
            "Tickets open for sale about 4 months in advance and rise in price closer to travel.",
        ],
    })


GENERAL_SNIPPETS = [
    "{topic}: a complete guide for first-time visitors, covering costs, timing and common mistakes. "
    "Most travelers find that planning two or three anchor activities per day keeps the pace comfortable.",
    "Updated guide to {topic}. Prices have risen modestly this year; booking a few weeks ahead secures "
    "better options, especially on weekends and public holidays.",
    "Forum thread: experiences with {topic}. Several travelers recommend comparing the official site with "
    "third-party resellers, and reading cancellation terms carefully before paying.",
    "{topic} explained: what is included, what costs extra, and how to save money. Local transit cards "
    "usually pay for themselves within two days of regular use.",
    "Ten things to know about {topic}. Crowds peak late morning, so arriving at opening time or in the "
    "last two hours before closing gives a calmer experience.",
    "Is it worth it? An honest review of {topic}, with a cost breakdown, time required, accessibility "
    "notes, and alternatives for travelers on a tighter budget.",
]
SITES = ["travelnotes.example", "wanderguide.example", "citybrief.example", "tripforum.example",
         "budgetroam.example", "localinsider.example"]


def _general(query: str) -> str:
    r = D.rng_for("web", query)
    topic = query.strip().rstrip("?")
    results = []
    for i, tpl in enumerate(r.sample(GENERAL_SNIPPETS, k=6)):
        site = SITES[i % len(SITES)]
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60]
        results.append({
            "rank": i + 1,
            "title": f"{topic.title()} | {site.split('.')[0].title()}",
            "url": f"https://{site}/{slug}",
            "snippet": tpl.format(topic=topic),
            "published": f"2026-{r.randint(1, 9):02d}-{r.randint(1, 28):02d}",
        })
    return _dump({"query": query, "type": "web_results", "results": results})


def web_search(query: str) -> str:
    """Search the web. Flight and train queries return structured results."""
    q = query.lower()
    if "train" in q or "rail" in q or "shinkansen" in q:
        return _trains(query)
    if "flight" in q or "fly" in q or "airfare" in q:
        return _flights(query)
    return _general(query)


# ---------------------------------------------------------------------------
# places_search: hotels, restaurants, attractions
# ---------------------------------------------------------------------------

HOTEL_WORDS_RE = re.compile(r"hotel|stay|resort|hostel|accommodation|lodging|room|villa|b&b|guesthouse")
FOOD_WORDS_RE = re.compile(r"restaurant|dinner|lunch|breakfast|food|eat|cafe|bar|brunch|dining")


def _category(query: str) -> str:
    q = query.lower()
    if HOTEL_WORDS_RE.search(q):
        return "hotels"
    if FOOD_WORDS_RE.search(q):
        return "restaurants"
    return "attractions"


def _gen_hotels(city: str, area: str | None, r) -> list[dict]:
    areas = D.areas_of(city)
    base = {"japan": 150, "italy": 170, "switzerland": 230, "france": 190, "netherlands": 180,
            "portugal": 120, "indonesia": 90}.get(D.country_of(city), 160)
    out = []
    for _ in range(8):
        kind, mult = r.choice(D.HOTEL_KINDS)
        a = area or r.choice(areas)
        out.append({
            "name": f"{r.choice(D.HOTEL_WORDS)} {a} {kind}",
            "area": a,
            "price_per_night_usd": int(base * mult * r.uniform(0.7, 1.5)),
            "stars": max(1, min(5, round(2 + mult * 1.8 + r.uniform(-0.6, 0.6)))),
            "description": f"{kind} in {a} with {r.choice(['modern', 'classic', 'minimalist', 'cozy'])} rooms, "
                           f"{r.randint(2, 12)} minutes on foot from the nearest transit stop.",
        })
    return out


def _decorate_hotel(h: dict, city: str, r) -> dict:
    h = dict(h)
    h.setdefault("area", r.choice(D.areas_of(city)))
    h["rating"] = round(r.uniform(3.8, 4.9), 1)
    h["review_count"] = r.randint(120, 4800)
    h["amenities"] = r.sample(D.AMENITIES, k=6)
    h["check_in"] = "15:00"
    h["check_out"] = r.choice(["10:00", "11:00", "12:00"])
    h["cancellation"] = r.choice(["Free cancellation until 48h before arrival.",
                                  "Non-refundable rate; 10% cheaper than flexible.",
                                  "Free cancellation until 7 days before arrival, then first night charged."])
    h["recent_reviews"] = [t.format(n=r.randint(2, 6), area=h["area"]) for t in r.sample(D.REVIEW_TEMPLATES, k=2)]
    return h


def _gen_restaurants(city: str, area: str | None, r) -> list[dict]:
    flavor = D.COUNTRY_FLAVOR.get(D.country_of(city), D.COUNTRY_FLAVOR["usa"])
    out = []
    for _ in range(8):
        cuisine = r.choice(flavor["cuisines"])
        a = area or r.choice(D.areas_of(city))
        out.append({
            "name": f"{r.choice(flavor['words'])} {cuisine.title()}",
            "cuisine": cuisine,
            "area": a,
            "avg_price_usd": r.randint(12, 75),
            "signature_dishes": flavor["dishes"].get(cuisine, ["house special"]),
            "description": f"{cuisine.title()} spot in {a} known for {flavor['dishes'].get(cuisine, ['its menu'])[0]}. "
                           f"{r.choice(['Casual and loud.', 'Quiet and intimate.', 'Popular with locals.', 'Great for groups.'])}",
        })
    return out


def _decorate_restaurant(x: dict, city: str, area: str | None, r) -> dict:
    x = dict(x)
    x.setdefault("area", area or r.choice(D.areas_of(city)))
    x["rating"] = round(r.uniform(3.9, 4.8), 1)
    x["review_count"] = r.randint(80, 3900)
    x["price_level"] = "$" * max(1, min(4, x["avg_price_usd"] // 20 + 1))
    x["hours"] = r.choice(["11:30-14:30, 17:30-22:00", "17:00-23:30", "07:00-15:00", "12:00-22:00"])
    x["reservations"] = r.choice(["recommended", "walk-in only", "required for dinner"])
    dish = x["signature_dishes"][0]
    x["recent_reviews"] = [t.format(n=r.randint(10, 45), dish=dish) for t in r.sample(D.FOOD_REVIEWS, k=2)]
    return x


def _gen_attractions(city: str, area: str | None, query: str, r) -> list[dict]:
    q = query.lower()
    out: list[dict] = []
    if city == "lisbon":
        for name, a, price, hours in D.LISBON_ATTRACTIONS:
            out.append({"name": name, "area": a, "price_usd": price, "duration_hours": hours,
                        "description": f"{name} in {a}. A popular stop for first-time visitors; "
                                       f"plan about {hours:g} hours including travel time."})
        return out
    if city == "bali":
        out.extend(dict(a) for a in D.BALI_ANCHORS["activities"])
    areas = D.areas_of(city)
    for tpl in r.sample(D.ATTRACTION_TEMPLATES, k=8 - len(out)):
        a = area or r.choice(areas)
        name = tpl.format(area=a, city=city.title())
        if any(w in q for w in ("tour", "ticket", "lesson", "class", "cruise")):
            name = f"{name} ({query.strip()})" if r.random() < 0.3 else name
        out.append({
            "name": name, "area": a, "price_usd": r.choice([0, 0, 12, 18, 25, 35, 49, 65]),
            "duration_hours": r.choice([1.0, 1.5, 2.0, 3.0, 4.0]),
            "description": f"{name}: {r.choice(['a must-see', 'a local favourite', 'a hidden gem', 'good on rainy days'])}, "
                           f"best visited {r.choice(['early morning', 'late afternoon', 'at sunset', 'midweek'])}.",
        })
    return out


def _decorate_attraction(x: dict, r) -> dict:
    x = dict(x)
    x["rating"] = round(r.uniform(4.0, 4.9), 1)
    x["review_count"] = r.randint(200, 25000)
    x["opening_hours"] = r.choice(["09:00-18:00", "10:00-19:00", "08:30-17:30", "open 24h"])
    x["booking"] = r.choice(["Timed-entry tickets sell out on weekends; book ahead.",
                             "Tickets at the door; queues of 20-40 minutes at midday.",
                             "Free entry; donations welcome."])
    x["accessibility"] = r.choice(["Step-free access to main areas.", "Steep stairs, not wheelchair accessible.",
                                   "Partially accessible; elevator to upper level."])
    return x


def places_search(query: str, location: str) -> str:
    """Search hotels, restaurants or attractions near a location."""
    city, area = D.resolve_city(location)
    r = D.rng_for("places", query, location)
    cat = _category(query)
    loc_l = location.lower()

    if cat == "hotels":
        if city == "amalfi coast":
            items = [dict(h) for h in D.AMALFI_HOTELS]
        elif city == "bali":
            items = [dict(h) for h in D.BALI_ANCHORS["hotels"]] + _gen_hotels(city, area, r)[:6]
        else:
            items = _gen_hotels(city, area, r)
        items = [_decorate_hotel(h, city, r) for h in items]
    elif cat == "restaurants":
        if "tsukiji" in loc_l or "tsukiji" in query.lower():
            items = [dict(x, area="Tsukiji") for x in D.TSUKIJI_RESTAURANTS]
        else:
            items = _gen_restaurants(city, area, r)
        items = [_decorate_restaurant(x, city, area, r) for x in items]
    else:
        items = [_decorate_attraction(x, r) for x in _gen_attractions(city, area, query, r)]

    return _dump({
        "query": query,
        "location": location,
        "resolved_city": city.title(),
        "category": cat,
        "result_count": len(items),
        "results": items,
        "disclaimer": "Ratings and prices are indicative and may have changed since last update.",
    })


# ---------------------------------------------------------------------------
# weather_fetch
# ---------------------------------------------------------------------------

CLIMATE = {  # (base high C, spread, rain chance)
    "japan": (22, 5, 0.35), "italy": (26, 4, 0.15), "indonesia": (31, 2, 0.45),
    "switzerland": (15, 6, 0.35), "france": (20, 5, 0.3), "netherlands": (17, 4, 0.45),
    "portugal": (24, 4, 0.12), "usa": (19, 4, 0.2),
}
CONDITIONS = ["sunny", "partly cloudy", "cloudy", "light rain", "showers", "clear", "windy"]


def weather_fetch(location: str, days: int = 7) -> str:
    """Daily forecast with packing and activity advice."""
    city, _ = D.resolve_city(location)
    hi0, spread, rain = CLIMATE.get(D.country_of(city), (20, 5, 0.25))
    if city == "zermatt":
        hi0 -= 8
    r = D.rng_for("weather", city)
    out = []
    for d in range(max(1, min(int(days), 10))):
        hi = hi0 + r.randint(-spread, spread)
        lo = hi - r.randint(6, 11)
        cond = "light rain" if r.random() < rain else r.choice(CONDITIONS)
        out.append({
            "day": f"Day {d + 1}",
            "condition": cond,
            "high_c": hi, "low_c": lo,
            "high_f": round(hi * 9 / 5 + 32), "low_f": round(lo * 9 / 5 + 32),
            "precipitation_chance": f"{r.randint(5, 30) if 'rain' not in cond else r.randint(55, 90)}%",
            "humidity": f"{r.randint(40, 85)}%",
            "wind_kmh": r.randint(5, 35),
            "uv_index": r.randint(2, 10),
            "summary": f"Morning {r.choice(['cool', 'mild', 'humid', 'crisp'])}, afternoon {cond}, "
                       f"evening {r.choice(['pleasant', 'breezy', 'damp', 'clear'])}.",
        })
    return _dump({
        "location": location,
        "resolved_city": city.title(),
        "units": "metric (imperial included)",
        "forecast": out,
        "packing_advice": [
            "Layers: mornings and evenings are noticeably cooler than afternoons.",
            "A compact umbrella or light rain jacket for showers.",
            "Comfortable walking shoes; expect 15,000+ steps on sightseeing days.",
        ],
        "activity_advice": "Schedule outdoor sightseeing for the driest days and keep museums or "
                           "indoor markets as backups for rainy afternoons.",
    })


# ---------------------------------------------------------------------------
# budget_tracker (stateful, per conversation)
# ---------------------------------------------------------------------------

class BudgetTracker:
    """Running spend tally. It only tracks spending, not the user's budget cap:
    remembering the cap is the agent's job (that is what scenario B tests)."""

    def __init__(self) -> None:
        self.expenses: list[tuple[str, float]] = []

    def __call__(self, action: str, item: str | None = None, amount: float | None = None) -> str:
        action = (action or "").lower().strip()
        if action in ("add", "add_expense"):
            if item is None or amount is None:
                return _dump({"error": "add_expense needs item and amount"})
            self.expenses.append((str(item), float(amount)))
            status = "expense recorded"
        elif action in ("remove", "remove_expense"):
            before = len(self.expenses)
            self.expenses = [e for e in self.expenses if e[0].lower() != str(item).lower()]
            status = "expense removed" if len(self.expenses) < before else f"no expense named '{item}'"
        elif action in ("summary", "get_summary", "total", "get_total"):
            status = "summary"
        elif action == "reset":
            self.expenses = []
            status = "ledger cleared"
        else:
            return _dump({"error": f"unknown action '{action}'",
                          "actions": ["add_expense", "remove_expense", "get_summary", "reset"]})
        total = sum(a for _, a in self.expenses)
        lines = [f"{i + 1}. {name}: ${amt:,.2f}" for i, (name, amt) in enumerate(self.expenses)]
        return (f"BUDGET TRACKER: {status}\n"
                f"ledger:\n" + ("\n".join(lines) if lines else "(empty)") + "\n"
                f"TOTAL SPENT: ${total:,.2f} ({len(self.expenses)} expense{'s' if len(self.expenses) != 1 else ''})")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_registry() -> ToolRegistry:
    """Fresh registry for one conversation (new budget ledger)."""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        "web_search", "Search the web for flights, trains and general travel information.",
        {"query": "string"}, web_search))
    reg.register(ToolSpec(
        "places_search", "Find hotels, restaurants or attractions near a location.",
        {"query": "string", "location": "string"}, places_search))
    reg.register(ToolSpec(
        "weather_fetch", "Get the daily weather forecast for a location.",
        {"location": "string", "days": "integer (optional, default 7)"}, weather_fetch))
    reg.register(ToolSpec(
        "budget_tracker", "Record expenses and get the running total spent. "
                          "Actions: add_expense, remove_expense, get_summary, reset.",
        {"action": "string", "item": "string (optional)", "amount": "number (optional)"},
        BudgetTracker()))
    return reg
