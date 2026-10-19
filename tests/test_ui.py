import json
import threading
import time
import unittest
import urllib.request
from functools import partial

from ctxcomp.context import make_context
from ctxcomp.eval import load_scenarios
from ctxcomp.llm import LLMBackend, LLMOutput
from ctxcomp.messages import ToolCall
from ctxcomp.ui import server as ui_server
from ctxcomp.ui.server import build_payload, render_page
from ctxcomp.ui.trace import trace_scenario


class Fake(LLMBackend):
    def generate(self, messages, **hints):
        if hints.get("step") == 0:
            return LLMOutput(ToolCall("places_search", {"query": "hotels", "location": "Rome"}).to_text(), 5, 5)
        return LLMOutput("Here are some hotels in Rome.", 5, 5)


class TestTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = {s.id[0]: s for s in load_scenarios()}

    def test_fact_status_over_time(self):
        run = trace_scenario(self.sc["A"], partial(make_context, "naive", max_context_tokens=8192))
        self.assertEqual(run["turns"][0]["facts"]["shellfish_allergy"], "in")
        self.assertEqual(run["turns"][-1]["facts"]["shellfish_allergy"], "lost")
        self.assertEqual(run["fact_source"]["shellfish_allergy"], 0)
        self.assertFalse(run["summary"]["task_pass"])
        # prompt indices always point at real messages from this turn or earlier
        last = run["turns"][-1]
        for i in last["ctx"]:
            self.assertLessEqual(run["messages"][i]["turn"], last["t"])

    def test_stale_status_respects_cancellation_turn(self):
        run = trace_scenario(self.sc["C"], partial(make_context, "naive", max_context_tokens=1_000_000))
        self.assertEqual(run["turns"][3]["stale"]["bali_research"], "current")
        self.assertEqual(run["turns"][-1]["stale"]["bali_research"], "leaked")

    def test_trace_matches_harness_summary(self):
        from ctxcomp.eval import run_scenario
        for strategy, ctx in (("naive", 16384), ("compressed", 2048)):
            f = partial(make_context, strategy, max_context_tokens=ctx)
            for k in self.sc:
                self.assertEqual(trace_scenario(self.sc[k], f)["summary"]["task_pass"],
                                 run_scenario(self.sc[k], f).task_pass, f"{strategy} {k}")

    def test_compressed_trace_marks_compacted_history(self):
        run = trace_scenario(self.sc["A"], partial(make_context, "compressed", max_context_tokens=8192))
        last = run["turns"][-1]
        kinds = {x["kind"] for x in last["ctx"] if isinstance(x, dict)}
        self.assertIn("ledger", kinds)
        self.assertIn("digest", kinds)
        self.assertTrue(any("src" in x for x in last["ctx"] if isinstance(x, dict)))
        self.assertIn("shellfish", last["memory"])
        self.assertTrue(last["decision"])

    def test_new_session_is_recorded(self):
        run = trace_scenario(self.sc["I"], partial(make_context, "compressed", max_context_tokens=8192))
        self.assertEqual([t["session"] for t in run["turns"]][-2:], [2, 2])
        self.assertEqual(run["turns"][-1]["facts"]["vegan"], "in")


class TestExport(unittest.TestCase):
    def test_render_page_embeds_data(self):
        payload = build_payload(load_scenarios(only=["A"]), [("naive", 8192)])
        html = render_page(dict(payload, mode="static"))
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("window.__CTXCOMP__={", html)
        self.assertNotIn("<!--CTXCOMP_DATA-->", html)
        fragment = render_page(payload, standalone=False)
        self.assertFalse(fragment.lstrip().startswith("<!doctype"))


class TestServer(unittest.TestCase):
    def test_live_session_flow(self):
        payload = build_payload(load_scenarios(only=["A"]), [("naive", 8192)])
        port = 8799
        t = threading.Thread(target=ui_server.serve, kwargs=dict(
            payload=payload, port=port, backend=Fake(), open_browser=False,
            live_info={"backend": "fake", "model": "fake", "strategies": ["naive"], "contexts": [8192]}), daemon=True)
        t.start()
        time.sleep(0.5)
        base = f"http://127.0.0.1:{port}/"

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())

        with urllib.request.urlopen(base) as r:
            page = r.read().decode()
        self.assertIn('"available":true', page)
        sid = post("api/session", {"strategy": "naive", "max_ctx": 8192})["id"]
        run = post("api/send", {"id": sid, "text": "Find hotels in Rome"})["run"]
        self.assertEqual(len(run["turns"]), 1)
        self.assertEqual(run["turns"][0]["tools"][0]["name"], "places_search")
        self.assertEqual(run["turns"][0]["reply"], "Here are some hotels in Rome.")


if __name__ == "__main__":
    unittest.main()
