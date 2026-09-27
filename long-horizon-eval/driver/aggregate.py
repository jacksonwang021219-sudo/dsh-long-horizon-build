"""Aggregate benchmark run reports into the comparison table.

    python3 aggregate.py --runs ../runs/full-standard.json ../runs/full-long-horizon.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    """Read one benchmark run report."""
    return json.loads(path.read_text())


def totals(report: dict) -> dict[str, object]:
    """Token and behaviour totals for one run.

    Each per-turn entry reports the session's CUMULATIVE statistics at the end
    of that turn, so a total is the last entry, never their sum; only wall
    clock is per-turn and therefore additive.
    """
    turns = report.get("turns", [])
    last = turns[-1] if turns else {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
             "cache_write_tokens": 0, "reasoning_tokens": 0, "calls": 0, "total_tokens": 0}
    for key in usage:
        usage[key] = int((last.get("usage") or {}).get(key, 0) or 0)
    return {
        "usage": usage,
        "steps": int(last.get("steps", 0) or 0),
        "tool_calls": int(last.get("tool_calls", 0) or 0),
        "records": int(last.get("records", 0) or 0),
        "folds": int(last.get("folds", 0) or 0),
        "seconds": round(sum(float(turn.get("seconds", 0) or 0) for turn in turns), 1),
    }


def scores(report: dict, key: str) -> dict[str, object]:
    """The grader's summary for one pass."""
    summary = report.get(key) or {}
    return {
        "total": summary.get("total"),
        "max": summary.get("max"),
        "attempted": summary.get("attempted"),
        "passes": summary.get("passes"),
        "rows": summary.get("subtasks"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    reports = [load(Path(path)) for path in args.runs]

    lines = ["| metric | " + " | ".join(report["mode"] for report in reports) + " |",
             "| --- | " + " | ".join("---" for _ in reports) + " |"]
    rows: list[tuple[str, list[str]]] = []

    def add(label: str, values: list[str]) -> None:
        rows.append((label, values))

    add("preset", [str(report["preset"]) for report in reports])
    for key, label in (("pass1", "score pass 1 (total/attempted)"),
                       ("pass2", "score pass 2 (total/attempted)")):
        values = []
        for report in reports:
            summary = scores(report, key)
            values.append(
                f"{summary['total']}/{summary['max']}" if summary["total"] is not None else "—"
            )
        add(label, values)
    add("subtasks passing (>=6/10), pass 1",
        [str(scores(report, "pass1")["passes"] or 0) for report in reports])
    add("subtasks passing (>=6/10), pass 2",
        [str(scores(report, "pass2")["passes"] or 0) for report in reports])
    add("user turns", [str(len(report.get("turns", []))) for report in reports])
    add("agent steps", [str(totals(report)["steps"]) for report in reports])
    add("tool calls", [str(totals(report)["tool_calls"]) for report in reports])
    add("records written", [str(totals(report)["records"]) for report in reports])
    add("window folds", [str(totals(report)["folds"]) for report in reports])
    add("wall clock (s)", [str(totals(report)["seconds"]) for report in reports])
    for key, label in (("total_tokens", "total tokens"),
                       ("input_tokens", "fresh input tokens"),
                       ("cache_read_tokens", "cache-read tokens"),
                       ("output_tokens", "output tokens"),
                       ("reasoning_tokens", "reasoning tokens"),
                       ("calls", "model calls")):
        add(label, [f"{totals(report)['usage'][key]:,}" for report in reports])
    add("tokens per step",
        [f"{totals(report)['usage']['total_tokens'] / max(1, totals(report)['steps']):,.0f}"
         for report in reports])

    for label, values in rows:
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "modes": [report["mode"] for report in reports],
            "table": table,
            "detail": [
                {
                    "mode": report["mode"],
                    "session_id": report.get("session_id"),
                    "totals": totals(report),
                    "pass1": scores(report, "pass1"),
                    "pass2": scores(report, "pass2"),
                    "turns": [
                        {k: turn.get(k) for k in ("label", "seconds", "steps", "tool_calls", "records", "folds", "usage")}
                        for turn in report.get("turns", [])
                    ],
                }
                for report in reports
            ],
        }, ensure_ascii=False, indent=2))
        print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
