"""Command line interface.

Examples:
  # Deterministic replay (no model needed): baseline vs compression module
  python -m ctxcomp eval --strategies naive compressed --max-context 8192

  # Ablation table: which component matters at which window size
  python -m ctxcomp ablate --max-context 8192 3072 2048

  # Live run with a local model through Ollama
  python -m ctxcomp eval --mode live --backend ollama --model qwen2.5:7b --strategies naive compressed

  # Context Inspector web UI (add --backend ollama --model ... for live chat)
  python -m ctxcomp ui

  # Talk to the agent in the terminal
  python -m ctxcomp chat --backend ollama --model qwen2.5:7b --strategy compressed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

from .agent import TravelAgent
from .context import REGISTRY, make_context
from .context.compressed import ABLATIONS
from .context.extract import LLMExtractor
from .eval import load_scenarios, run_scenario
from .eval.report import ablation_table, markdown_report, summary_table
from .llm import make_backend

UNLIMITED = 1_000_000
DEFAULT_UI_RUNS = "naive:8192,naive:16384,naive:unlimited,compressed:8192,compressed:2048"


def _ctx_value(v: str) -> int:
    return UNLIMITED if str(v).lower() in ("unlimited", "inf", "none") else int(v)


def _ctx_label(c: int) -> str:
    return "unlimited" if c >= UNLIMITED else str(c)


def _backend_from_args(args):
    if args.backend in ("scripted", "replay"):
        return None
    kw = {"model": args.model}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.backend == "ollama":
        kw["num_ctx"] = min(max(args.max_context), 131072)
    if args.api_key:
        kw["api_key"] = args.api_key
    return make_backend(args.backend, **kw)


def _factory(args, backend):
    """Returns make(strategy, max_ctx) -> ContextManager, honoring --extractor."""
    extractor = None
    if getattr(args, "extractor", "heuristic") == "llm":
        if backend is None:
            sys.exit("--extractor llm needs a model (--backend ollama or openai)")
        extractor = LLMExtractor(backend)

    def make(strategy: str, max_ctx: int):
        kw = {"max_context_tokens": max_ctx, "reserve_output_tokens": args.reserve_output}
        if extractor is not None:
            kw["extractor"] = extractor
        return make_context(strategy, **kw)
    return make


def _run_all(args, strategies, contexts, backend, verbose=True):
    scenarios = load_scenarios(args.scenario_dir, args.scenarios)
    if not scenarios:
        sys.exit("No scenarios matched.")
    make = _factory(args, backend)
    results = []
    for strategy in strategies:
        for ctx in contexts:
            for sc in scenarios:
                r = run_scenario(sc, partial(make, strategy, ctx), backend=backend, max_steps=args.max_steps)
                results.append(r)
                if verbose:
                    lost = [k for k, v in r.retained.items() if not v]
                    leaked = [k for k, v in r.stale_leaked.items() if v]
                    print(f"[{'PASS' if r.task_pass else 'FAIL'}] {sc.id:26s} {strategy:27s} ctx={_ctx_label(ctx):>9} "
                          f"retention={r.retention_rate:4.0%} lost={lost or '-'} leaked={leaked or '-'} "
                          f"prompt_tokens={r.total_prompt_tokens:,}")
    return results


def _save(args, results, tag, extra_md=""):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    (out / f"{name}.json").write_text(json.dumps([r.to_dict() for r in results], indent=2))
    (out / f"{name}.md").write_text(markdown_report(results, f"Evaluation: {tag}") + extra_md)
    print(f"\nSaved {out / name}.json and .md")


def cmd_eval(args) -> int:
    backend = _backend_from_args(args) if args.mode == "live" else None
    results = _run_all(args, args.strategies, args.max_context, backend)
    print()
    print(summary_table(results))
    _save(args, results, f"{args.mode}_{'-'.join(args.strategies)}")
    return 0


def cmd_ablate(args) -> int:
    strategies = ["naive", "compressed"] + [cls.name for cls in ABLATIONS]
    backend = _backend_from_args(args) if args.mode == "live" else None
    results = _run_all(args, strategies, args.max_context, backend, verbose=False)
    table = ablation_table(results)
    print(table)
    _save(args, results, f"ablation_{args.mode}", "\n\n## Ablation\n\n" + table + "\n")
    return 0


def cmd_chat(args) -> int:
    backend = _backend_from_args(args)
    if backend is None:
        print("chat needs a real backend (--backend ollama or openai).", file=sys.stderr)
        return 2
    ctx = _factory(args, backend)(args.strategy, max(args.max_context))
    agent = TravelAgent(backend, ctx, max_steps=args.max_steps)
    print(f"Travel agent ready ({args.strategy}, {max(args.max_context)} tokens). Commands: /stats, /memory, "
          f"/newsession, quit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text == "/stats":
            print(json.dumps(agent.context.stats(), indent=2, default=str))
            continue
        if text == "/memory":
            ledger = getattr(agent.context, "render_ledger", None)
            print(ledger() if ledger else "(this strategy keeps no separate memory)")
            continue
        if text == "/newsession":
            agent.context = agent.context.new_session()
            print("(new session started; persistent memory carried over if the strategy has any)")
            continue
        res = agent.chat(text)
        for c in res.tool_calls:
            print(f"  [tool] {c.name}({json.dumps(c.args)})")
        print(f"agent> {res.final_text}")
        print(f"  ({res.llm_calls[-1].prompt_tokens:,} prompt tokens)")
    return 0


def _parse_runs(spec: str) -> list[tuple[str, int]]:
    runs = []
    for item in spec.split(","):
        strategy, _, ctx = item.strip().partition(":")
        if strategy not in REGISTRY:
            sys.exit(f"unknown strategy '{strategy}' in --runs. Available: {sorted(REGISTRY)}")
        runs.append((strategy, _ctx_value(ctx or "8192")))
    return runs


def cmd_ui(args) -> int:
    from .ui.server import build_payload, render_page, serve

    scenarios = load_scenarios(args.scenario_dir, args.scenarios)
    runs = _parse_runs(args.runs)
    args.max_context = [max(c for _, c in runs)]   # size the model's window for the largest run
    backend = _backend_from_args(args)
    trace_backend = backend if args.trace_live else None
    make = _factory(args, backend)
    print(f"Tracing {len(scenarios)} scenarios x {len(runs)} runs ({'live' if trace_backend else 'replay'})...")
    payload = build_payload(scenarios, runs, make, trace_backend, progress=args.verbose)
    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_page(dict(payload, mode="static")), encoding="utf-8")
        print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB). Open it in any browser.")
        return 0
    live_info = None
    if backend is not None:
        live_info = {"backend": args.backend, "model": args.model,
                     "strategies": sorted({s for s, _ in runs}),
                     "contexts": sorted({c for _, c in runs})}
    serve(payload, host=args.host, port=args.port, backend=backend, live_info=live_info, make_context=make,
          max_steps=args.max_steps, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ctxcomp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    scen_dir = str(Path(__file__).resolve().parents[1] / "scenarios")

    def common(sp, default_ctx=(8192,)):
        sp.add_argument("--backend", default="scripted", choices=["scripted", "ollama", "openai"])
        sp.add_argument("--model", default="qwen2.5:7b")
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--api-key", default=None)
        sp.add_argument("--max-context", nargs="+", type=_ctx_value, default=list(default_ctx),
                        help="context window(s) in tokens; 'unlimited' for no limit")
        sp.add_argument("--reserve-output", type=int, default=512)
        sp.add_argument("--max-steps", type=int, default=5, help="max tool calls per user turn")
        sp.add_argument("--extractor", default="heuristic", choices=["heuristic", "llm"],
                        help="how the compression module extracts facts (llm needs a live backend)")

    def scen(sp):
        sp.add_argument("--scenarios", nargs="+", default=["all"], help="ids or letters, e.g. A C")
        sp.add_argument("--scenario-dir", default=scen_dir)

    e = sub.add_parser("eval", help="run scenarios and report")
    common(e)
    scen(e)
    e.add_argument("--mode", default="replay", choices=["replay", "live"])
    e.add_argument("--strategies", nargs="+", default=["naive", "compressed"], choices=sorted(REGISTRY))
    e.add_argument("--out", default="results")
    e.set_defaults(func=cmd_eval)

    a = sub.add_parser("ablate", help="turn each compression component off and compare")
    common(a, default_ctx=(8192, 3072, 2048))
    scen(a)
    a.add_argument("--mode", default="replay", choices=["replay", "live"])
    a.add_argument("--out", default="results")
    a.set_defaults(func=cmd_ablate)

    c = sub.add_parser("chat", help="interactive session with the agent")
    common(c)
    c.add_argument("--strategy", default="compressed", choices=sorted(REGISTRY))
    c.set_defaults(func=cmd_chat)

    u = sub.add_parser("ui", help="Context Inspector web UI (or --export a static page)")
    common(u, default_ctx=(8192,))
    scen(u)
    u.add_argument("--runs", default=DEFAULT_UI_RUNS,
                   help=f"comma-separated strategy:context pairs (default {DEFAULT_UI_RUNS})")
    u.add_argument("--trace-live", action="store_true",
                   help="run the scenario traces with the live model instead of replay (slow)")
    u.add_argument("--export", default=None, help="write a self-contained HTML file and exit")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")
    u.add_argument("--verbose", action="store_true")
    u.set_defaults(func=cmd_ui)

    args = p.parse_args(argv)
    if getattr(args, "mode", None) == "live" and args.backend == "scripted":
        p.error("--mode live needs --backend ollama or openai")
    if args.cmd == "ui" and args.trace_live and args.backend == "scripted":
        p.error("--trace-live needs --backend ollama or openai")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
