"""End-to-end verification of the sliding-window policy against a live harness.

Runs a few cheap user turns under a preset whose thresholds are small enough to
exercise every fold rule (``lh-stress``), then reads the durable log back and
reports what the window actually did: records written, folds performed, tokens
spent, and whether any turn closed without a record.

    python3 verify_window.py --base-url http://127.0.0.1:58422 --preset lh-stress
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness, stats_from_events  # noqa: E402

SHORT_TURNS = [
    "Create a file named notes-1.txt in the working directory containing exactly the line: alpha. Then reply with only the file name.",
    "Create a file named notes-2.txt containing exactly the line: beta. Then reply with only the file name.",
    "Create a file named notes-3.txt containing exactly the line: gamma. Then reply with only the file name.",
    "Create a file named notes-4.txt containing exactly the line: delta. Then reply with only the file name.",
]

# One turn big enough to cross a small in-turn token trigger: the agent writes a
# large file with a shell loop and then reads it back in pieces.
LONG_TURN = (
    "In the working directory, run a bash loop that writes a file named bulk.txt with 4000 lines, "
    "line i reading 'row i <i>' followed by forty filler words. Then report the file's line count and "
    "the text of line 3999. Work only through tools, and do not paste the whole file into your answer."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--preset", default="lh-stress")
    parser.add_argument("--workdir", default=str(Path.home() / "lh-window-verify"))
    parser.add_argument("--turns", type=int, default=len(SHORT_TURNS))
    parser.add_argument("--long-turn", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs" / "verify-window.json"))
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    harness = Harness(args.base_url)
    preset_ids = [item.get("id") for item in harness.presets()]
    if args.preset not in preset_ids:
        print(f"[FAIL] preset {args.preset!r} not available; roster: {preset_ids}")
        return 1

    session_id = harness.create_session(str(workdir), args.preset)
    print(f"[INFO] session {session_id} on preset {args.preset} in {workdir}")

    prompts = list(SHORT_TURNS[: args.turns])
    if args.long_turn:
        prompts.append(LONG_TURN)

    for index, prompt in enumerate(prompts, start=1):
        started = time.time()
        harness.prompt(session_id, prompt)
        idle = harness.wait_idle(session_id, timeout=args.timeout)
        events = harness.read_log(session_id)
        stats = stats_from_events(events, session_id)
        print(
            f"[TURN {index}] idle={idle} {time.time() - started:6.1f}s "
            f"turns={stats.turns} steps={stats.steps} tools={stats.tool_calls} "
            f"records={stats.records} folds={stats.folds} "
            f"in={stats.usage.input_tokens} cache={stats.usage.cache_read_tokens} "
            f"out={stats.usage.output_tokens}"
        )
        if not idle:
            print("[WARN] turn did not settle within the timeout")

    events = harness.read_log(session_id)
    stats = stats_from_events(events, session_id)
    folds = [
        {
            "seq": event.get("seq"),
            "range": (event.get("data") or {}).get("shadowedRange"),
            "tokens": (event.get("data") or {}).get("shadowedTokenCount"),
        }
        for event in events
        if event.get("type") == "compaction/prune"
    ]
    blocks = [
        {
            "seq": event.get("seq"),
            "kind": "turn-record" if "<turn-record" in json.dumps(event.get("data", {})) else "folded-history",
        }
        for event in events
        if event.get("type") == "user/message"
        and isinstance(event.get("data"), dict)
        and (event["data"].get("source") or {}).get("plugin") == "dsh-context-window"
    ]
    report = {
        "session_id": session_id,
        "preset": args.preset,
        "stats": stats.as_dict(),
        "folds": folds,
        "window_blocks": blocks,
        "log_path": str(harness.log_path(session_id)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("stats", "folds", "window_blocks")}, ensure_ascii=False, indent=2))
    print(f"[OK] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
