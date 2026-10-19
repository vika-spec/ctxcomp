import json
import unittest
from functools import partial

from ctxcomp.agent import TravelAgent
from ctxcomp.context import CompressedContext, NaiveContext, make_context
from ctxcomp.context.digest import digest
from ctxcomp.context.extract import LLMExtractor, classify, find_pivots
from ctxcomp.eval import load_scenarios, run_scenario
from ctxcomp.llm import LLMBackend, LLMOutput, ScriptedLLM
from ctxcomp.messages import ToolCall
from ctxcomp.tokens import count_message_tokens, count_tokens
from ctxcomp.tools import make_registry

SCENARIOS = {s.id[0]: s for s in load_scenarios()}


def play(letter, strategy="compressed", ctx=8192):
    """Replay a scenario and return (agent, context) at the end."""
    llm = ScriptedLLM()
    context = make_context(strategy, max_context_tokens=ctx)
    agent = TravelAgent(llm, context, make_registry())
    for t in SCENARIOS[letter].turns:
        if t.new_session:
            agent.context = agent.context.new_session()
        llm.set_plan(t.tools, t.reply)
        agent.chat(t.user)
    return agent, agent.context


class TestExtraction(unittest.TestCase):
    def test_constraint_kinds(self):
        self.assertEqual(classify("I'm severely allergic to shellfish.")[0], "allergy")
        self.assertEqual(classify("She has celiac disease, so nothing with gluten for her.")[0], "allergy")
        self.assertEqual(classify("Our budget is $1,800 total."), ["budget"])
        self.assertEqual(classify("My dad uses a wheelchair, so no stairs please."), ["mobility"])
        self.assertIn("appointment", classify("I have a meeting in Paris on Wednesday at 2pm."))
        self.assertIn("pace", classify("Max 2 activities per day."))
        self.assertIn("itinerary", classify("I land in Rome on Tuesday morning."))

    def test_questions_and_bookings_do_not_set_budgets(self):
        self.assertEqual(classify("Can we afford a $250 kaiseki dinner for two?"), [])
        self.assertEqual(classify("Book the round trip, it comes to $800 total."), [])
        self.assertEqual(classify("Keep hotels under $200 a night."), ["price_limit"])

    def test_requests_are_not_preferences(self):
        self.assertEqual(classify("Find hotels in Rome."), [])
        self.assertEqual(classify("What's the weather in Kyoto?"), [])

    def test_pivots(self):
        self.assertEqual(find_pivots("Actually, scratch Bali entirely."), ["Bali"])
        self.assertEqual(find_pivots("Let's do Norway instead of Santorini."), ["Santorini"])
        self.assertEqual(find_pivots("Forget it, I'll decide later."), [])

    def test_llm_extractor_rejects_paraphrases(self):
        class Fake(LLMBackend):
            def generate(self, messages, **hints):
                return LLMOutput(json.dumps([
                    {"kind": "diet", "quote": "I eat no meat at all"},            # paraphrase: rejected
                    {"kind": "preference", "quote": "love quiet old towns"},      # verbatim: accepted
                ]), 1, 1)
        facts = LLMExtractor(Fake())("We love quiet old towns. I'm vegetarian.", 1)
        texts = [f.text for f in facts]
        self.assertIn("love quiet old towns", texts)
        self.assertNotIn("I eat no meat at all", texts)
        self.assertIn("I'm vegetarian.", texts)   # heuristic result kept


class TestDigest(unittest.TestCase):
    def test_places_digest_is_small_but_complete(self):
        raw = make_registry().call("places_search", {"query": "hotels", "location": "Shinjuku, Tokyo"})
        d = digest("places_search", raw)
        self.assertLess(count_tokens(d) * 5, count_tokens(raw))
        for r in json.loads(raw)["results"]:
            self.assertIn(r["name"], d)

    def test_short_outputs_kept(self):
        out = make_registry().call("budget_tracker", {"action": "get_summary"})
        self.assertIn("TOTAL SPENT", digest("budget_tracker", out))


class TestCompressedBehaviour(unittest.TestCase):
    def test_budget_math_in_memory(self):
        _, ctx = play("B")
        mem = ctx.render_ledger()
        self.assertIn("cap $2,500", mem)
        self.assertIn("remaining $950", mem)

    def test_budget_raise_supersedes_old_cap(self):
        agent, ctx = play("G")
        mem = ctx.render_ledger()
        self.assertIn("cap $2,400", mem)
        self.assertIn("remaining $380", mem)
        prompt = "\n".join(m.content for m in ctx.build(agent.system_prompt))
        self.assertNotIn("$1,800", prompt)

    def test_pivot_invalidates_cancelled_plan(self):
        agent, ctx = play("C")
        self.assertEqual(sorted(ctx.stale_turns), [1, 2, 3, 4, 5, 6])
        prompt = "\n".join(m.content for m in ctx.build(agent.system_prompt))
        for word in ("Bali", "Seminyak", "Ubud", "Kuta"):
            self.assertNotIn(word, prompt)
        self.assertIn("mountains, not beaches", prompt)

    def test_allergy_survives_a_pivot(self):
        ctx = CompressedContext()
        ctx.on_user("Plan a beach week in Bali. I'm allergic to peanuts.", 1)
        ctx.on_assistant("Sure!", 1)
        ctx.on_user("Actually, scratch Bali entirely. Let's do Norway instead.", 2)
        mem = ctx.render_ledger()
        self.assertIn("allergic to peanuts", mem)
        self.assertNotIn("Bali", mem)

    def test_duplicate_results_are_deduplicated(self):
        _, ctx = play("E")
        dups = [e for r in ctx.turns.values() for e in r.calls if e.dup_of]
        self.assertGreater(len(dups), 0)

    def test_session_state_round_trip(self):
        _, ctx = play("A")
        state = json.loads(json.dumps(ctx.to_state()))
        fresh = CompressedContext()
        fresh.load_state(state)
        self.assertEqual(fresh.session, 2)
        self.assertIn("allergic to shellfish", fresh.render_ledger())
        self.assertIsInstance(NaiveContext().new_session(), NaiveContext)


class TestScenarioResults(unittest.TestCase):
    def test_compressed_passes_everything_at_8k_and_2k(self):
        for ctx in (8192, 2048):
            for letter, sc in SCENARIOS.items():
                r = run_scenario(sc, partial(make_context, "compressed", max_context_tokens=ctx))
                self.assertTrue(r.task_pass, f"{letter} at {ctx}: lost {r.retained} leaked {r.stale_leaked}")

    def test_baseline_fails_everything_at_8k(self):
        for letter, sc in SCENARIOS.items():
            r = run_scenario(sc, partial(make_context, "naive", max_context_tokens=8192))
            self.assertFalse(r.task_pass, letter)

    def test_prompt_never_exceeds_the_window(self):
        for strategy in ("compressed", "compressed-no-ledger", "compressed-no-digests", "naive"):
            for ctx in (8192, 2048, 1536):
                for letter, sc in SCENARIOS.items():
                    r = run_scenario(sc, partial(make_context, strategy, max_context_tokens=ctx))
                    self.assertLessEqual(r.peak_prompt_tokens, ctx - 512, f"{strategy} {letter} {ctx}")

    def test_compression_saves_tokens(self):
        sc = SCENARIOS["E"]
        base = run_scenario(sc, partial(make_context, "naive", max_context_tokens=1_000_000))
        comp = run_scenario(sc, partial(make_context, "compressed", max_context_tokens=8192))
        self.assertLess(comp.total_prompt_tokens * 4, base.total_prompt_tokens)


if __name__ == "__main__":
    unittest.main()
