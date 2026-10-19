"""Synthetic travel data.

Everything here is fictional sample data (hotel, restaurant and airline names,
prices, schedules). Public landmarks keep their real names, but their prices
and hours are made up. The data is deterministic: the same query always returns
the same payload, so evaluation runs are reproducible.

A few "anchor" entries are hand-written because the evaluation scenarios depend
on them (seafood-heavy restaurants in Tsukiji, a luxury resort on the Amalfi
Coast, a long list of Lisbon attractions). Everything else is generated.
"""

from __future__ import annotations

import hashlib
import random

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

# city -> (country key, areas)
CITIES: dict[str, tuple[str, list[str]]] = {
    "tokyo": ("japan", ["Shinjuku", "Shibuya", "Asakusa", "Ginza", "Tsukiji", "Ueno"]),
    "kyoto": ("japan", ["Gion", "Higashiyama", "Arashiyama", "Kyoto Station", "Pontocho"]),
    "osaka": ("japan", ["Namba", "Umeda", "Shinsekai", "Dotonbori"]),
    "rome": ("italy", ["Trastevere", "Monti", "Centro Storico", "Prati", "Termini"]),
    "florence": ("italy", ["Duomo", "Oltrarno", "Santa Croce", "San Lorenzo"]),
    "amalfi coast": ("italy", ["Positano", "Amalfi", "Ravello", "Praiano"]),
    "bali": ("indonesia", ["Seminyak", "Ubud", "Kuta", "Canggu", "Uluwatu"]),
    "zurich": ("switzerland", ["Altstadt", "Kreis 4", "Seefeld", "Enge"]),
    "zermatt": ("switzerland", ["Village Centre", "Winkelmatten", "Steinmatte"]),
    "interlaken": ("switzerland", ["Hoeheweg", "Unterseen", "Matten"]),
    "lucerne": ("switzerland", ["Old Town", "Lakeside", "Tribschen"]),
    "geneva": ("switzerland", ["Old Town", "Paquis", "Eaux-Vives"]),
    "paris": ("france", ["Eiffel Tower", "Le Marais", "Saint-Germain", "Montmartre", "Latin Quarter"]),
    "amsterdam": ("netherlands", ["Jordaan", "Centrum", "De Pijp", "Museumplein", "Oost"]),
    "lisbon": ("portugal", ["Alfama", "Baixa", "Belem", "Bairro Alto", "Chiado"]),
    "san francisco": ("usa", ["Union Square", "Mission", "SoMa"]),
    "new york": ("usa", ["Midtown", "SoHo", "Brooklyn"]),
}

# Aliases a user or model might type.
CITY_ALIASES = {
    "amalfi": "amalfi coast", "positano": "amalfi coast", "ravello": "amalfi coast",
    "praiano": "amalfi coast", "seminyak": "bali", "ubud": "bali", "kuta": "bali",
    "canggu": "bali", "uluwatu": "bali", "tsukiji": "tokyo", "shinjuku": "tokyo",
    "asakusa": "tokyo", "shibuya": "tokyo", "ginza": "tokyo", "gion": "kyoto",
    "kansai": "osaka", "sintra": "lisbon", "cascais": "lisbon", "belem": "lisbon",
    "alfama": "lisbon", "trastevere": "rome", "sfo": "san francisco", "nyc": "new york",
    "switzerland": "zurich", "italy": "rome", "japan": "tokyo", "france": "paris",
    "netherlands": "amsterdam", "portugal": "lisbon", "indonesia": "bali",
}

AIRPORTS = {
    "tokyo": "HND", "kyoto": "KIX", "osaka": "KIX", "rome": "FCO", "florence": "FLR",
    "amalfi coast": "NAP", "bali": "DPS", "zurich": "ZRH", "zermatt": "ZRH",
    "interlaken": "ZRH", "lucerne": "ZRH", "geneva": "GVA", "paris": "CDG",
    "amsterdam": "AMS", "lisbon": "LIS", "san francisco": "SFO", "new york": "JFK",
}

COUNTRY_FLAVOR = {
    "japan": {
        "cuisines": ["ramen", "yakitori", "tempura", "tonkatsu", "udon", "sushi", "izakaya",
                     "okonomiyaki", "soba", "curry rice"],
        "words": ["Hanabi", "Kaze", "Tsuki", "Sakura", "Matsu", "Yuki", "Hoshi", "Kumo", "Hikari", "Mori"],
        "dishes": {
            "ramen": ["tonkotsu ramen", "shoyu ramen", "gyoza"],
            "yakitori": ["negima skewers", "tsukune meatballs", "chicken thigh skewers"],
            "tempura": ["vegetable tempura", "shrimp tempura", "tempura don"],
            "tonkatsu": ["pork loin katsu", "katsu curry", "shredded cabbage"],
            "udon": ["kake udon", "kitsune udon", "curry udon"],
            "sushi": ["omakase nigiri", "tuna sashimi", "sweet shrimp nigiri"],
            "izakaya": ["karaage", "edamame", "grilled squid"],
            "okonomiyaki": ["pork okonomiyaki", "yakisoba", "seafood okonomiyaki"],
            "soba": ["zaru soba", "duck soba", "tempura soba"],
            "curry rice": ["beef curry rice", "katsu curry", "vegetable curry"],
        },
    },
    "italy": {
        "cuisines": ["trattoria", "pizzeria", "osteria", "seafood", "enoteca", "gelateria", "pasta bar"],
        "words": ["Da Mario", "Il Sole", "La Pergola", "Al Vicolo", "Del Ponte", "La Stella", "Il Gatto", "Dei Fiori"],
        "dishes": {
            "trattoria": ["cacio e pepe", "saltimbocca", "carciofi alla romana"],
            "pizzeria": ["margherita", "diavola", "pizza bianca"],
            "osteria": ["pappardelle al ragu", "bistecca", "ribollita"],
            "seafood": ["spaghetti alle vongole", "fritto misto", "grilled octopus"],
            "enoteca": ["cheese board", "salumi plate", "bruschetta"],
            "gelateria": ["pistachio gelato", "lemon sorbet", "affogato"],
            "pasta bar": ["carbonara", "amatriciana", "fresh tagliatelle"],
        },
    },
    "indonesia": {
        "cuisines": ["warung", "seafood grill", "balinese", "cafe", "vegan cafe"],
        "words": ["Warung Made", "Sari", "Bunga", "Laut", "Dewi", "Surya", "Padi"],
        "dishes": {
            "warung": ["nasi campur", "mie goreng", "sate ayam"],
            "seafood grill": ["grilled prawns", "jimbaran fish", "calamari"],
            "balinese": ["babi guling", "bebek betutu", "lawar"],
            "cafe": ["smoothie bowl", "avocado toast", "flat white"],
            "vegan cafe": ["tempeh bowl", "jackfruit tacos", "raw cake"],
        },
    },
    "switzerland": {
        "cuisines": ["fondue", "raclette", "alpine bistro", "rosti house", "chocolate cafe"],
        "words": ["Edelweiss", "Gletscher", "Alpenblick", "Matterhorn", "Bergli", "Seeblick"],
        "dishes": {
            "fondue": ["moitie-moitie fondue", "fondue chinoise", "dried meat platter"],
            "raclette": ["raclette with potatoes", "cornichons", "air-dried beef"],
            "alpine bistro": ["alplermagronen", "veal sausage", "apple strudel"],
            "rosti house": ["rosti with egg", "zurcher geschnetzeltes", "bircher muesli"],
            "chocolate cafe": ["hot chocolate", "truffles", "nusstorte"],
        },
    },
    "france": {
        "cuisines": ["bistro", "brasserie", "creperie", "patisserie", "wine bar"],
        "words": ["Le Petit", "Chez Paul", "La Belle", "Le Comptoir", "Au Vieux", "Le Jardin"],
        "dishes": {
            "bistro": ["steak frites", "duck confit", "onion soup"],
            "brasserie": ["choucroute", "moules frites", "croque monsieur"],
            "creperie": ["galette complete", "crepe suzette", "cider"],
            "patisserie": ["croissant", "eclair", "tarte au citron"],
            "wine bar": ["charcuterie", "comte", "natural wine flight"],
        },
    },
    "netherlands": {
        "cuisines": ["brown cafe", "rijsttafel", "pancake house", "bistro", "herring stand"],
        "words": ["De Gouden", "Het Hoekje", "Café 't", "De Blauwe", "Van Dam", "De Kroon"],
        "dishes": {
            "brown cafe": ["bitterballen", "uitsmijter", "apple pie"],
            "rijsttafel": ["beef rendang", "sate", "gado-gado"],
            "pancake house": ["bacon pancake", "poffertjes", "stroopwafel"],
            "bistro": ["stamppot", "fish of the day", "kale salad"],
            "herring stand": ["raw herring", "kibbeling", "smoked eel"],
        },
    },
    "portugal": {
        "cuisines": ["tasca", "seafood", "pastelaria", "petiscos bar", "fado restaurant"],
        "words": ["O Cantinho", "A Taberna", "Casa do", "O Velho", "Tasca da", "A Brasa"],
        "dishes": {
            "tasca": ["bacalhau a bras", "bifana", "caldo verde"],
            "seafood": ["clams bulhao pato", "grilled sardines", "seafood rice"],
            "pastelaria": ["pastel de nata", "bola de berlim", "galao"],
            "petiscos bar": ["chourico assado", "pica-pau", "olives"],
            "fado restaurant": ["fado dinner menu", "polvo a lagareiro", "port wine"],
        },
    },
    "usa": {
        "cuisines": ["diner", "steakhouse", "taqueria", "bakery"],
        "words": ["Golden Gate", "Bay", "Mission", "Union"],
        "dishes": {
            "diner": ["burger", "pancakes", "milkshake"],
            "steakhouse": ["ribeye", "creamed spinach", "wedge salad"],
            "taqueria": ["burrito", "al pastor tacos", "horchata"],
            "bakery": ["sourdough", "morning bun", "cold brew"],
        },
    },
}

HOTEL_WORDS = ["Lumen", "Aster", "Harbor", "Meridian", "Willow", "Granite", "Juniper", "Solace",
               "Cobalt", "Ember", "Linden", "Halcyon"]
HOTEL_KINDS = [("Hotel", 1.0), ("Boutique Hotel", 1.4), ("Suites", 1.3), ("Inn", 0.8),
               ("Hostel", 0.35), ("Residence", 1.1), ("Guesthouse", 0.6)]
AMENITIES = ["free wifi", "24h front desk", "rooftop bar", "breakfast buffet", "laundry service",
             "luggage storage", "fitness room", "air conditioning", "airport shuttle",
             "late checkout on request", "in-room kettle", "blackout curtains", "family rooms",
             "accessible rooms", "bicycle rental", "concierge desk"]
REVIEW_TEMPLATES = [
    "Stayed {n} nights and the staff remembered our names by day two. {area} location made it easy to walk everywhere.",
    "Rooms are on the small side but spotless. Breakfast had a good mix of local and western options.",
    "Great value for the price. A bit of street noise at night, earplugs were provided which helped.",
    "Lovely views from the upper floors. Ask for a high room, the lower ones face the courtyard.",
    "Check-in was slow because of a tour group, otherwise everything was smooth and friendly.",
    "Would book again. The concierge gave us excellent tips for local food and transit passes.",
    "Beds were very comfortable, shower pressure excellent. Elevator is tiny but works fine.",
    "Perfect for a short stay. Close to transit, convenience stores and a couple of good cafes.",
]
FOOD_REVIEWS = [
    "We waited about {n} minutes for a table but it was worth it. Portions are generous.",
    "Friendly staff and a menu in English. The {dish} was the highlight of our trip.",
    "Small place, maybe twelve seats, so go early. Prices are fair for the quality.",
    "A bit touristy but the food is honest and fresh. Service was quick even when busy.",
    "Locals were eating here which is always a good sign. Try the {dish}.",
    "Cash only when we visited. Cozy atmosphere and the chef chatted with us at the counter.",
]
ATTRACTION_TEMPLATES = [
    "{area} Walking Tour", "{area} Food Market", "{city} Museum of Art", "{area} Viewpoint",
    "{city} History Museum", "{area} Night Market", "{city} Botanical Garden", "{area} Cooking Class",
    "{city} River Cruise", "{area} Bike Tour", "{city} Old Town Square", "{area} Photo Walk",
]

# ---------------------------------------------------------------------------
# Hand-written anchors used by the evaluation scenarios
# ---------------------------------------------------------------------------

TSUKIJI_RESTAURANTS = [
    {"name": "Uogashi Sushi Counter", "cuisine": "sushi (seafood)", "avg_price_usd": 65,
     "signature_dishes": ["sea urchin nigiri", "sweet shrimp nigiri", "scallop nigiri", "crab miso soup"],
     "description": "Ten-seat counter inside the outer market run by a third-generation itamae. The omakase "
                    "leans heavily on the morning catch: sweet shrimp, scallops, sea urchin and seasonal crab."},
    {"name": "Tsukiji Kaisen-don Ichiba", "cuisine": "seafood rice bowls", "avg_price_usd": 32,
     "signature_dishes": ["kaisen-don with prawns and crab", "salmon roe bowl", "grilled scallop skewer"],
     "description": "Bustling rice-bowl shop famous for overflowing kaisen-don piled with prawns, crab, "
                    "salmon roe and tuna. Expect a queue at lunch, dinner service is calmer."},
    {"name": "Hamayaki Shellfish Grill", "cuisine": "seafood grill", "avg_price_usd": 48,
     "signature_dishes": ["grilled oysters", "butter-soy scallops", "whole grilled lobster tail", "clam sake broth"],
     "description": "Charcoal grill where shellfish is cooked at your table: oysters, scallops, clams and "
                    "lobster tails. Smoky, lively and very popular after dark."},
    {"name": "Torisen Yakitori", "cuisine": "yakitori (chicken)", "avg_price_usd": 30,
     "signature_dishes": ["negima skewers", "tsukune with egg yolk", "chicken oyster cut", "grilled shiitake"],
     "description": "Binchotan-grilled chicken skewers, a short sake list and a relaxed counter. The menu is "
                    "almost entirely chicken and vegetables."},
    {"name": "Katsu Hanabi", "cuisine": "tonkatsu (pork cutlet)", "avg_price_usd": 22,
     "signature_dishes": ["kurobuta pork loin katsu", "katsu curry", "miso soup with tofu"],
     "description": "Crisp panko pork cutlets from heritage-breed pigs, served with unlimited cabbage and rice."},
    {"name": "Kome to Yasai", "cuisine": "vegetable kaiseki", "avg_price_usd": 70,
     "signature_dishes": ["seasonal vegetable course", "sesame tofu", "clay-pot rice with mushrooms"],
     "description": "Quiet eight-course vegetable-forward kaiseki. Dashi is made from kombu and shiitake "
                    "on request; ask the staff when booking."},
    {"name": "Soba Kaze Tsukiji", "cuisine": "soba", "avg_price_usd": 18,
     "signature_dishes": ["zaru soba", "duck soba", "shrimp tempura soba"],
     "description": "Hand-cut buckwheat noodles in a tiny shop near the outer market. Tempura is fried in "
                    "shared oil with the shrimp."},
    {"name": "Maruyama Tamagoyaki", "cuisine": "street food", "avg_price_usd": 6,
     "signature_dishes": ["sweet rolled omelette on a stick", "dashi omelette sandwich"],
     "description": "Market stall selling rolled omelettes since 1951. Good for a snack between dinner stops."},
]

AMALFI_HOTELS = [
    {"name": "Grand Hotel Villa Aurelia", "price_per_night_usd": 520, "stars": 5,
     "description": "Cliffside luxury resort with infinity pool, private beach club and Michelin-starred terrace "
                    "dining. Minimum stay of three nights in high season."},
    {"name": "Palazzo Ravello Suites", "price_per_night_usd": 410, "stars": 5,
     "description": "Restored 12th-century palazzo above the coast with formal gardens and a spa."},
    {"name": "Hotel Belvedere Positano", "price_per_night_usd": 340, "stars": 4,
     "description": "Terraced hotel with sea-view balconies on every room and a shuttle to the beach."},
    {"name": "Casa Limoni Guesthouse", "price_per_night_usd": 145, "stars": 3,
     "description": "Family-run guesthouse among lemon groves in Praiano, homemade breakfast on the terrace."},
    {"name": "B&B La Scalinata", "price_per_night_usd": 120, "stars": 3,
     "description": "Four simple rooms up a staircase from Amalfi's main square. Great value, many steps."},
    {"name": "Ostello Mare Amalfi", "price_per_night_usd": 55, "stars": 2,
     "description": "Clean hostel with private rooms and dorms, popular with solo travelers, rooftop view."},
    {"name": "Hotel Marina Riviera", "price_per_night_usd": 260, "stars": 4,
     "description": "Harborfront hotel steps from the ferry pier, rooftop pool and lemon-themed bar."},
    {"name": "Villa Praiano Rooms", "price_per_night_usd": 175, "stars": 3,
     "description": "Bright rooms with shared panoramic terrace, ten minutes by bus from Positano."},
]

LISBON_ATTRACTIONS = [
    ("Belem Tower", "Belem", 12, 1.5), ("Jeronimos Monastery", "Belem", 14, 2.0),
    ("MAAT Museum", "Belem", 11, 1.5), ("Belem Pastry Tasting", "Belem", 25, 1.0),
    ("Sao Jorge Castle", "Alfama", 15, 2.0), ("Alfama Walking Tour", "Alfama", 20, 2.5),
    ("Fado Show in Alfama", "Alfama", 45, 2.0), ("National Tile Museum", "Alfama", 8, 1.5),
    ("Tram 28 Ride", "Baixa", 3, 1.0), ("Time Out Market Food Tour", "Baixa", 55, 2.5),
    ("Miradouro da Senhora do Monte", "Graca", 0, 0.5), ("LX Factory", "Alcantara", 0, 2.0),
    ("Gulbenkian Museum", "Avenidas Novas", 14, 2.5), ("Lisbon Oceanarium", "Parque das Nacoes", 25, 2.5),
    ("Pena Palace day trip (Sintra)", "Sintra", 40, 6.0), ("Quinta da Regaleira (Sintra)", "Sintra", 15, 2.5),
    ("Cascais Beach Afternoon", "Cascais", 6, 4.0), ("Sunset Sail on the Tagus", "Docas", 50, 2.0),
    ("Bairro Alto Bar Crawl", "Bairro Alto", 30, 3.0), ("Port Wine Tasting", "Chiado", 35, 1.5),
]

BALI_ANCHORS = {
    "hotels": [
        {"name": "Samudra Beach Resort Seminyak", "price_per_night_usd": 210, "stars": 5,
         "description": "Beachfront resort with three pools, sunset bar and direct beach access in Seminyak."},
        {"name": "Ombak Surf Villas Kuta", "price_per_night_usd": 95, "stars": 3,
         "description": "Villas two minutes from the surf break, board storage and wax station."},
    ],
    "activities": [
        {"name": "Kuta Surf Academy beginner lesson", "price_usd": 40, "duration_hours": 2.0,
         "description": "Small-group surf lessons on the beach break, soft boards and rash guards included."},
        {"name": "Ubud Water Temple Tour", "price_usd": 35, "duration_hours": 5.0,
         "description": "Guided tour of water temples and rice terraces around Ubud with a traditional lunch."},
        {"name": "Uluwatu Cliff Temple and Kecak Dance", "price_usd": 30, "duration_hours": 4.0,
         "description": "Sunset at the clifftop temple followed by the kecak fire dance."},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rng_for(*parts: object) -> random.Random:
    """Deterministic RNG seeded from the inputs (stable across processes)."""
    key = "|".join(str(p).lower().strip() for p in parts)
    seed = int(hashlib.md5(key.encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def resolve_city(text: str) -> tuple[str, str | None]:
    """Return (city_key, area) for a free-text location like 'Shinjuku, Tokyo'."""
    t = (text or "").lower()
    area = None
    for city, (_, areas) in CITIES.items():
        for a in areas:
            if a.lower() in t:
                area = a
                return city, area
    for city in sorted(CITIES, key=len, reverse=True):
        if city in t:
            return city, area
    for alias, city in CITY_ALIASES.items():
        if alias in t:
            areas = CITIES[city][1]
            match = next((a for a in areas if a.lower() == alias), None)
            return city, match
    return (t.split(",")[0].strip() or "unknown"), area


def country_of(city: str) -> str:
    return CITIES.get(city, ("usa", []))[0]


def areas_of(city: str) -> list[str]:
    return CITIES.get(city, ("usa", [city.title()]))[1] or [city.title()]
