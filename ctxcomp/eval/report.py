"""Console and Markdown reports."""

from __future__ import annotations

from collections import defaultdict

from .harness import ScenarioResult


def _ctx_label(n: int) -> str:
    return "unlimited" if n >= 1_000_000 else f"{n // 1024}K" if n % 1024 == 0 else str(n)


def _yn(v: bool | None) -> str:
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def results_table(results: list[ScenarioResult]) -> str:
    head = ("| Scenario | Strategy | Context | Task | Retained facts | Stale leaked | "
            "Prompt tokens (total) | Peak prompt | Msgs dropped |")
    sep = "|" + "---|" * 9
    rows = [head, sep]
    for r in results:
        kept = ", ".join(f"{k}{'' if v else ' (LOST)'}" for k, v in r.retained.items()) or "-"
        leaked = ", ".join(k for k, v in r.stale_leaked.items() if v) or "none"
        rows.append(
            f"| {r.scenario_id} | {r.strategy} | {_ctx_label(r.max_context_tokens)} | {_yn(r.task_pass)} | "
            f"{kept} | {leaked} | {r.total_prompt_tokens:,} | {r.peak_prompt_tokens:,} | "
            f"{r.context_stats.get('dropped_messages_last_build', '-')} |")
    return "\n".join(rows)


def summary_table(results: list[ScenarioResult]) -> str:
    groups: dict[tuple[str, int], list[ScenarioResult]] = defaultdict(list)
    for r in results:
        groups[(r.strategy, r.max_context_tokens)].append(r)
    rows = ["| Strategy | Context | Task pass | Fact retention | Stale leakage | Mean prompt tokens/call | Total prompt tokens |",
            "|" + "---|" * 7]
    for (strategy, ctx), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(rs)
        passed = sum(r.task_pass for r in rs)
        retention = sum(r.retention_rate for r in rs) / n
        leaks = sum(any(r.stale_leaked.values()) for r in rs if r.stale_leaked)
        stale_n = sum(1 for r in rs if r.stale_leaked)
        mean_tok = sum(r.mean_prompt_tokens for r in rs) / n
        total = sum(r.total_prompt_tokens for r in rs)
        rows.append(f"| {strategy} | {_ctx_label(ctx)} | {passed}/{n} | {retention:.0%} | "
                    f"{leaks}/{stale_n} | {mean_tok:,.0f} | {total:,} |")
    return "\n".join(rows)


def ablation_table(results: list[ScenarioResult]) -> str:
    """Rows: strategies. Columns: context windows. Cells: scenarios passed (total prompt tokens)."""
    contexts = sorted({r.max_context_tokens for r in results}, reverse=True)
    order = []
    for r in results:
        if r.strategy not in order:
            order.append(r.strategy)
    head = "| Strategy | " + " | ".join(_ctx_label(c) for c in contexts) + " |"
    rows = [head, "|" + "---|" * (len(contexts) + 1)]
    for s in order:
        cells = []
        for c in contexts:
            rs = [r for r in results if r.strategy == s and r.max_context_tokens == c]
            if not rs:
                cells.append("-")
                continue
            passed = sum(r.task_pass for r in rs)
            leaks = sum(any(r.stale_leaked.values()) for r in rs)
            tok = sum(r.total_prompt_tokens for r in rs)
            cells.append(f"{passed}/{len(rs)}{f', {leaks} leak' + ('s' if leaks > 1 else '') if leaks else ''} "
                         f"({tok / 1000:,.0f}k tok)")
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def markdown_report(results: list[ScenarioResult], title: str) -> str:
    mode = results[0].mode if results else "?"
    parts = [f"# {title}", "",
             f"Mode: **{mode}**. " + ("Task pass = the final answer passed its checks." if mode == "live" else
                                       "Replay mode: task pass = every critical fact was visible to the model "
                                       "when it wrote the final answer, and no cancelled information was."),
             "", "## Summary", "", summary_table(results), "", "## Per scenario", "", results_table(results), ""]
    if mode == "live":
        parts += ["## Final answers", ""]
        for r in results:
            parts += [f"### {r.scenario_id} ({r.strategy}, {_ctx_label(r.max_context_tokens)}): {_yn(r.answer_pass)}",
                      "", "> " + r.final_answer.replace("\n", "\n> "), ""]
    errs = [(r.scenario_id, e) for r in results for e in r.errors]
    if errs:
        parts += ["## Errors", ""] + [f"- {sid}: {e}" for sid, e in errs]
    return "\n".join(parts)
