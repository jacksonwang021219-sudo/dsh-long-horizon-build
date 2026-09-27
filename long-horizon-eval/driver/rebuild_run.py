"""Rebuild a run report's token/step totals from the session log itself.

Runs that failed *after* their agent turns (for example while grading) leave no
report JSON, but their durable session log still holds the complete accounting.
This finds the session by its workspace path and writes the same totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import Harness, stats_from_events  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:58700")
    parser.add_argument("--run-root", required=True, help="the run's workspace root (session cwd)")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--base-url-for-session", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=0.0, help="wall clock measured by the batch runner")
    args = parser.parse_args()

    harness = Harness(args.base_url_for_session or args.base_url)
    target = str(Path(args.run_root).resolve())
    found = None
    for item in harness.summaries():
        sid = item.get("sessionId")
        if not isinstance(sid, str):
            continue
        try:
            events = harness.read_log(sid)
        except Exception:  # noqa: BLE001
            continue
        header = next((e for e in events if e.get("type") == "session"), {})
        if str(header.get("cwd") or "") == target:
            found = (sid, events)
            break
    if found is None:
        print(f"[MISS] no session for {target}")
        return 1
    sid, events = found
    stats = stats_from_events(events, sid)
    report = {
        "mode": args.mode,
        "model_name": args.model_name,
        "session_id": sid,
        "run_root": target,
        "rebuilt_from_log": True,
        "turns": [{
            "label": "all",
            "seconds": args.seconds,
            "steps": stats.steps,
            "tool_calls": stats.tool_calls,
            "records": stats.records,
            "folds": stats.folds,
            "usage": stats.usage.as_dict(),
        }],
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {args.model_name}: tokens={stats.usage.total_tokens:,} steps={stats.steps} "
          f"records={stats.records} folds={stats.folds} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
