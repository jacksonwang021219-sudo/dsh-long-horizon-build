"""Deep-dive one benchmark session: per-turn behaviour, folds, and records.

    python3 analyze.py --session session-<id> [--base-url URL]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness, stats_from_events, text_of_content  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:58422")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    harness = Harness(args.base_url)
    events = harness.read_log(args.session)
    stats = stats_from_events(events, args.session)

    # Turn for every event: assistant/tool events carry it; a user/message takes
    # the turn opened most recently before it.
    turn_of_seq: dict[int, int] = {}
    current = 0
    for event in events:
        seq = event.get("seq")
        if not isinstance(seq, int):
            continue
        if event.get("type") == "turn/start":
            current = int((event.get("data") or {}).get("turn") or current)
        turn_of_seq[seq] = current

    per_turn: dict[int, dict[str, object]] = collections.defaultdict(
        lambda: {"steps": 0, "tools": collections.Counter(), "tokens": 0, "out": 0, "cache": 0}
    )
    folds: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    notices: list[dict[str, object]] = []
    for event in events:
        seq = event.get("seq")
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        turn = turn_of_seq.get(seq if isinstance(seq, int) else -1, 0)
        if kind == "step/start":
            per_turn[turn]["steps"] = int(per_turn[turn]["steps"]) + 1
        elif kind == "tool/call":
            name = str(data.get("name"))
            per_turn[turn]["tools"][name] += 1
            arguments = str(data.get("arguments") or "")
            if len(arguments) > 400:
                per_turn[turn].setdefault("big_calls", []).append((name, len(arguments)))
        elif kind == "assistant/message":
            usage = data.get("usage") or {}
            per_turn[turn]["tokens"] = int(per_turn[turn]["tokens"]) + int(usage.get("inputTokens", 0) or 0) \
                + int(usage.get("cacheReadTokens", 0) or 0) + int(usage.get("outputTokens", 0) or 0)
            per_turn[turn]["out"] = int(per_turn[turn]["out"]) + int(usage.get("outputTokens", 0) or 0)
        elif kind == "compaction/prune":
            folds.append({
                "seq": seq,
                "turn": turn,
                "range": data.get("shadowedRange"),
                "tokens": data.get("shadowedTokenCount"),
                "nodes": len(data.get("shadowedSeqs") or []),
            })
        elif kind == "tool/result":
            meta = data.get("meta")
            if isinstance(meta, dict) and meta.get("kind") == "dsh-context-window/record":
                records.append({
                    "seq": seq,
                    "turn": meta.get("turn"),
                    "recordKind": meta.get("recordKind"),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                    "tokens": meta.get("tokens"),
                    "summary_chars": len(str(meta.get("summary") or "")),
                })
        elif kind == "user/message":
            source = data.get("source") or {}
            if source.get("plugin") == "dsh-context-window":
                text = text_of_content(data.get("content"))
                notices.append({
                    "seq": seq,
                    "turn": turn,
                    "kind": "notice" if "window-notice" in text else "block",
                    "chars": len(text),
                })

    report = {
        "session": args.session,
        "stats": stats.as_dict(),
        "turns": [
            {
                "turn": turn,
                "steps": per_turn[turn]["steps"],
                "tokens": per_turn[turn]["tokens"],
                "output_tokens": per_turn[turn]["out"],
                "tools": dict(per_turn[turn]["tools"]),
            }
            for turn in sorted(per_turn)
        ],
        "folds": folds,
        "records": records,
        "window_blocks": notices,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
