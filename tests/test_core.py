import json
import unittest
from functools import partial

from ctxcomp.agent import TravelAgent
from ctxcomp.context import NaiveContext, make_context
from ctxcomp.eval import load_scenarios, run_scenario
from ctxcomp.eval.checks import run_answer_check
from ctxcomp.llm import LLMBackend, LLMOutput, ScriptedLLM, parse_tool_call
from ctxcomp.messages import ToolCall
from ctxcomp.tokens import count_message_tokens, count_tokens
from ctxcomp.tools import make_registry


class TestTools(unittest.TestCase):
    def test_deterministic(self):
        a = make_registry().call("places_search", {"query": "hotels", "location": "Rome"})
        b = make_registry().call("places_search", {"query": "hotels", "location": "Rome"})
        self.assertEqual(a, b)

    def test_payloads_are_large_enough_to_stress_context(self):
        reg = make_registry()
        out = reg.call("places_search", {"query": "hotels", "location": "Shinjuku, Tokyo"})
        self.assertGreater(count_tokens(out), 1000)
        self.assertEqual(len(json.loads(out)["results"]), 8)

    def test_budget_tracker_running_total(self):
        reg = make_registry()
        reg.call("budget_tracker", {"action": "add_expense", "item": "flight", "amount": 800})
        reg.call("budget_tracker", {"action": "add_expense", "item": "hotel", "amount": 400})
        out = reg.call("budget_tracker", {"action": "get_summary"})
        self.assertIn("TOTAL SPENT: $1,200.00", out)
        out = reg.call("budget_tracker", {"action": "remove_expense", "item": "hotel"})
        self.assertIn("TOTAL SPENT: $800.00", out)

    def test_budget_is_per_conversation(self):
        make_registry().call("budget_tracker", {"action": "add_expense", "item": "x", "amount": 5})
        self.assertIn("$0.00", make_registry().call("budget_tracker", {"action": "get_summary"}))

    def test_bad_calls_return_errors_not_exceptions(self):
        reg = make_registry()
        self.assertIn("unknown tool", reg.call("nope", {}))
        self.assertIn("bad arguments", reg.call("weather_fetch", {"city": "Rome"}))

    def test_anchor_data(self):
        reg = make_registry()
        tsukiji = reg.call("places_search", {"query": "dinner", "location": "Tsukiji, Tokyo"})
        self.assertIn("Hamayaki Shellfish Grill", tsukiji)
        amalfi = reg.call("places_search", {"query": "hotels", "location": "Amalfi Coast"})
        self.assertIn("Villa Aurelia", amalfi)
        lisbon = json.loads(reg.call("places_search", {"query": "attractions", "location": "Lisbon"}))
        self.assertGreaterEqual(lisbon["result_count"], 15)


class TestParser(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_tool_call('<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'),
                         ToolCall("a", {"x": 1}))
        self.assertEqual(parse_tool_call('```json\n{"name": "b", "arguments": {}}\n```').name, "b")
        self.assertEqual(parse_tool_call('{"tool": "c", "args": {"q": "z"}}'), ToolCall("c", {"q": "z"}))
        self.assertEqual(parse_tool_call('<tool_call>{"name": "d", "arguments": "{\\"k\\": 2}"}</tool_call>').args,
                         {"k": 2})

    def test_plain_text_is_not_a_tool_call(self):
        self.assertIsNone(parse_tool_call("Here are three options: A, B and C."))
        self.assertIsNone(parse_tool_call("Use {braces} freely."))

    def test_roundtrip(self):
        call = ToolCall("places_search", {"query": "hotels", "location": "Rome"})
        self.assertEqual(parse_tool_call(call.to_text()), call)


class TestNaiveContext(unittest.TestCase):
    def test_keeps_everything_when_it_fits(self):
        ctx = NaiveContext(max_context_tokens=100_000)
        for t in range(1, 6):
            ctx.on_user(f"message {t}", t)
            ctx.on_assistant(f"reply {t}", t)
        msgs = ctx.build("sys")
        self.assertEqual(len(msgs), 11)
        self.assertEqual(ctx.stats()["dropped_messages_last_build"], 0)

    def test_drops_oldest_and_respects_budget(self):
        ctx = NaiveContext(max_context_tokens=600, reserve_output_tokens=100)
        for t in range(1, 30):
            ctx.on_user("x" * 200, t)
        msgs = ctx.build("system prompt")
        total = sum(count_message_tokens(m.content) for m in msgs)
        self.assertLessEqual(total, ctx.prompt_budget)
        self.assertEqual(msgs[-1].turn, 29)
        self.assertGreater(msgs[1].turn, 1)
        self.assertGreater(ctx.stats()["dropped_messages_last_build"], 0)

    def test_oversized_single_message_is_trimmed_not_lost(self):
        ctx = NaiveContext(max_context_tokens=300, reserve_output_tokens=50)
        ctx.on_user("y" * 10_000, 1)
        msgs = ctx.build("sys")
        self.assertEqual(len(msgs), 2)
        self.assertLessEqual(sum(count_message_tokens(m.content) for m in msgs), ctx.prompt_budget)


class FakeLiveBackend(LLMBackend):
    """Calls one tool on the first step of each turn, then answers."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, messages, **hints):
        if hints.get("step") == 0 and messages[-1].role == "user":
            return LLMOutput(ToolCall("weather_fetch", {"location": "Rome"}).to_text(), 10, 5)
        return LLMOutput(self.answer, 10, 5)


class TestAgent(unittest.TestCase):
    def test_tool_then_answer(self):
        agent = TravelAgent(FakeLiveBackend("Sunny all week."), NaiveContext())
        res = agent.chat("Weather in Rome?")
        self.assertEqual([c.name for c in res.tool_calls], ["weather_fetch"])
        self.assertEqual(res.final_text, "Sunny all week.")
        self.assertEqual(len(res.llm_calls), 2)

    def test_step_limit(self):
        class Looper(LLMBackend):
            def generate(self, messages, **hints):
                return LLMOutput(ToolCall("weather_fetch", {"location": "Rome"}).to_text(), 1, 1)

        res = TravelAgent(Looper(), NaiveContext(), max_steps=3).chat("loop")
        self.assertTrue(res.hit_step_limit)
        self.assertEqual(len(res.tool_calls), 3)


class TestChecks(unittest.TestCase):
    def test_forbid_unless(self):
        chk = {"type": "forbid_unless", "forbid": ["Uogashi"], "unless": ["shellfish"]}
        self.assertFalse(run_answer_check(chk, "Try Uogashi Sushi Counter!")["passed"])
        self.assertTrue(run_answer_check(chk, "Uogashi is all shellfish, so avoid it.")["passed"])
        self.assertTrue(run_answer_check(chk, "Torisen Yakitori is great.")["passed"])

    def test_forbid_and_require(self):
        self.assertFalse(run_answer_check({"type": "forbid", "patterns": [r"\bBali\b"]}, "Unlike Bali...")["passed"])
        self.assertTrue(run_answer_check({"type": "require_any", "patterns": ["Thursday"]}, "Go Thursday.")["passed"])


class TestHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = {s.id[0]: s for s in load_scenarios()}

    def test_all_scenarios_load(self):
        self.assertEqual(sorted(self.scenarios), list("ABCDEFGHIJ"))
        self.assertEqual(len(self.scenarios["C"].turns), 26)
        self.assertEqual(len(self.scenarios["A"].turns), 16)

    def test_baseline_loses_allergy_at_8k(self):
        r = run_scenario(self.scenarios["A"], partial(make_context, "naive", max_context_tokens=8192))
        self.assertFalse(r.retained["shellfish_allergy"])
        self.assertFalse(r.task_pass)
        self.assertLessEqual(r.peak_prompt_tokens, 8192)

    def test_unlimited_context_keeps_allergy(self):
        r = run_scenario(self.scenarios["A"], partial(make_context, "naive", max_context_tokens=1_000_000))
        self.assertTrue(r.retained["shellfish_allergy"])

    def test_unlimited_context_leaks_cancelled_plan(self):
        r = run_scenario(self.scenarios["C"], partial(make_context, "naive", max_context_tokens=1_000_000))
        self.assertTrue(r.stale_leaked["bali_research"])
        self.assertFalse(r.task_pass)

    def test_live_mode_scores_answer(self):
        backend = FakeLiveBackend("Avoid Hamayaki and Uogashi because of your shellfish allergy; try Torisen.")
        r = run_scenario(self.scenarios["A"], partial(make_context, "naive", max_context_tokens=1_000_000),
                         backend=backend)
        self.assertEqual(r.mode, "live")
        self.assertTrue(r.answer_pass)

    def test_replay_is_deterministic(self):
        f = partial(make_context, "naive", max_context_tokens=8192)
        a = run_scenario(self.scenarios["B"], f)
        b = run_scenario(self.scenarios["B"], f)
        self.assertEqual(a.prompt_tokens_per_turn, b.prompt_tokens_per_turn)


if __name__ == "__main__":
    unittest.main()
