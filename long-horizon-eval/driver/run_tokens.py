"""Sum the true model spend of one benchmark run.

A run's cost is not one session: the agent may delegate to subagents, and each
child makes its own model calls under the same working directory. This walks
every session whose cwd is the run root and totals them, marking the root
session separately from its children.

    python3 run_tokens.py --run-root <dir> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import Harness, stats_from_events  # noqa: E402


def collect(base_url: str, run_root: str) -> dict[str, object]:
    """Total every session the run opened under one workspace root."""
    harness = Harness(base_url)
    target = str(Path(run_root).resolve())
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
             "cache_write_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    steps = records = folds = turns = 0
    sessions: list[dict[str, object]] = []
    for item in harness.summaries():
        sid = item.get("sessionId")
        if not isinstance(sid, str):
            continue
        try:
            events = harness.read_log(sid)
        except Exception:  # noqa: BLE001
            continue
        header = next((e for e in events if e.get("type") == "session"), {})
        if str(header.get("cwd") or "") != target:
            continue
        stats = stats_from_events(events, sid)
        if stats.steps == 0:
            continue
        for key in usage:
            usage[key] += getattr(stats.usage, key)
        steps += stats.steps
        records += stats.records
        folds += stats.folds
        turns += stats.turns
        sessions.append({
            "session_id": sid,
            "origin": header.get("origin", "root"),
            "depth": header.get("delegationDepth", 0),
            "turns": stats.turns,
            "steps": stats.steps,
            "tokens": stats.usage.total_tokens,
        })
    sessions.sort(key=lambda row: -int(row["tokens"] or 0))
    return {"run_root": target, "usage": usage, "steps": steps, "records": records,
            "folds": folds, "turns": turns, "sessions": sessions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:58700")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = collect(args.base_url, args.run_root)
    print(json.dumps({k: result[k] for k in ("run_root", "usage", "steps", "records", "folds", "turns")},
                     ensure_ascii=False))
    print("sessions:", [(s["origin"], s["depth"], s["session_id"][:14], s["tokens"]) for s in result["sessions"]])
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
