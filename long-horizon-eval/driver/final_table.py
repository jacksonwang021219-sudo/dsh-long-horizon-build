"""Final option-C table: score, tokens, steps and window behaviour per scenario.

Scores come from each scenario's own `meta_eval.json`; spend comes from walking
every session the run opened under its workspace root (the agent may delegate to
subagents, and each child bills separately).

    python3 final_table.py --base-url http://127.0.0.1:58700
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_optionc import SCENARIOS, scenario_score  # noqa: E402
from run_tokens import collect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench" / "AgencyBench" / "AgencyBench-v2"
BATCH = ROOT / "runs" / "optionC"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:58700")
    parser.add_argument("--out", default=str(BATCH / "optionC-final.json"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for slug, (scenario, subtasks, family) in SCENARIOS.items():
        for preset, prefix in (("standard", "std"), ("long-horizon", "lh")):
            model_name = f"{prefix}-{slug}"
            run_root = BENCH / scenario / model_name
            score = scenario_score(slug, model_name)
            spend = collect(args.base_url, str(run_root)) if run_root.exists() else {}
            wall = 0.0
            report = BATCH / f"{model_name}.json"
            if report.exists():
                turns = (json.loads(report.read_text()).get("turns") or [])
                wall = sum(float(turn.get("seconds", 0) or 0) for turn in turns)
            rows.append({
                "slug": slug, "scenario": scenario, "family": family, "subtasks": subtasks,
                "preset": preset, "model_name": model_name,
                "score": score.get("score"), "per_subtask": score.get("per_subtask"),
                "tokens": (spend.get("usage") or {}).get("total_tokens"),
                "fresh": (spend.get("usage") or {}).get("input_tokens"),
                "cache": (spend.get("usage") or {}).get("cache_read_tokens"),
                "output": (spend.get("usage") or {}).get("output_tokens"),
                "steps": spend.get("steps"), "records": spend.get("records"),
                "folds": spend.get("folds"),
                "sessions": len(spend.get("sessions") or []),
                "seconds": round(wall, 1),
            })

    def pair(slug: str) -> tuple[dict, dict]:
        found = {row["preset"]: row for row in rows if row["slug"] == slug}
        return found.get("standard", {}), found.get("long-horizon", {})

    print("| scenario | 子任务 | 标准分 | 长程分 | 差 | 标准 token | 长程 token | 长程/标准 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for slug, (scenario, subtasks, _family) in SCENARIOS.items():
        std, lh = pair(slug)
        if std.get("score") is None and lh.get("score") is None:
            continue
        delta = None
        if isinstance(std.get("score"), (int, float)) and isinstance(lh.get("score"), (int, float)):
            delta = round(float(lh["score"]) - float(std["score"]), 2)
        ratio = None
        if std.get("tokens") and lh.get("tokens"):
            ratio = round(float(lh["tokens"]) / float(std["tokens"]), 2)
        fmt = lambda value: f"{float(value) / 1e6:,.1f} M" if isinstance(value, (int, float)) else "—"  # noqa: E731
        print(f"| {scenario} | {subtasks} | {std.get('score')} | {lh.get('score')} | {delta} | "
              f"{fmt(std.get('tokens'))} | {fmt(lh.get('tokens'))} | {ratio} |")

    print("\n| scenario | 标准 步/记录/折叠 | 长程 步/记录/折叠 | 标准 墙钟 | 长程 墙钟 | 标准 会话 | 长程 会话 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for slug, (scenario, _subtasks, _family) in SCENARIOS.items():
        std, lh = pair(slug)
        if not std and not lh:
            continue
        cell = lambda row: f"{row.get('steps', '—')}/{row.get('records', '—')}/{row.get('folds', '—')}"  # noqa: E731
        mins = lambda row: f"{float(row['seconds']) / 60:,.0f} min" if row.get("seconds") else "—"  # noqa: E731
        print(f"| {scenario} | {cell(std)} | {cell(lh)} | {mins(std)} | {mins(lh)} | "
              f"{std.get('sessions', '—')} | {lh.get('sessions', '—')} |")

    summary: dict[str, object] = {"rows": rows}
    for preset in ("standard", "long-horizon"):
        scored = [row for row in rows if row["preset"] == preset and isinstance(row.get("score"), (int, float))
                  and row.get("tokens")]
        if not scored:
            continue
        summary[preset] = {
            "scenarios": len(scored),
            "subtasks": sum(int(row["subtasks"]) for row in scored),
            "mean_score": round(sum(float(row["score"]) for row in scored) / len(scored), 3),
            "total_tokens": sum(int(row["tokens"]) for row in scored),
            "total_steps": sum(int(row["steps"] or 0) for row in scored),
            "total_seconds": round(sum(float(row["seconds"] or 0) for row in scored), 1),
            "records": sum(int(row.get("records") or 0) for row in scored),
            "folds": sum(int(row.get("folds") or 0) for row in scored),
        }
        summary[preset]["tokens_per_step"] = round(
            summary[preset]["total_tokens"] / max(1, summary[preset]["total_steps"]))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
