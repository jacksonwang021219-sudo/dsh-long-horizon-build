"""Recompute one arm's wall clock, tokens, steps, folds, and notes without the retries.

The runner's own numbers (`runs/dsh-*-r*.json`) count everything the session log
holds, including the turns that died with a `TRANSPORT` error and were resent.
This walks the durable log instead and splits it three ways:

* **completed turns** — a `turn/end` whose reason is not an error: this is the
  work that actually produced the scored deliverables;
* **error turns** — a `turn/end` with `reason.kind == 'error'`: the model stream
  broke, the work was thrown away and redone, so its tokens and minutes are the
  price of the interruption;
* **gaps** — time between one turn's end and the next one's start (grading runs,
  workspace copies, retry backoff).

Every other per-turn figure (tokens by cache class, steps, tool calls, folds,
notes written for that turn) is attributed by the turn the event's seq belongs to.

    python3 clean_metrics.py runs/dsh-v1-r1.json ... --out runs/clean-metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness  # noqa: E402

NOTES_ROOT = Path.home() / ".dsh" / "notes"


def decode(session_id: str) -> list[dict[str, object]]:
    """Decode one session's durable log into event dictionaries."""
    path = Harness("http://127.0.0.1:1").log_path(session_id)
    if path is None:
        raise SystemExit(f"no durable log for {session_id}")
    completed = subprocess.run(
        ["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
        check=True, capture_output=True, text=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def usage_of(message: dict[str, object]) -> dict[str, int]:
    """The provider accounting one `assistant/message` carries."""
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    return {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
        "cache_read_tokens": int(usage.get("cacheReadTokens") or 0),
    }


def money(tokens: dict[str, int]) -> float:
    """Off-peak price of one token bucket, in CNY."""
    return (
        tokens["input_tokens"] * 1.0
        + tokens["cache_read_tokens"] * 0.02
        + tokens["output_tokens"] * 4.0
    ) / 1_000_000


def analyse(path: Path) -> dict[str, object]:
    """Split one arm's log into completed work and the price of the interruptions."""
    report = json.loads(path.read_text())
    events = decode(str(report["session_id"]))
    turns: dict[int, dict[str, object]] = {}
    order: list[int] = []
    current: int | None = None
    previous_end: int | None = None
    gap_ms = 0
    for event in events:
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        time = event.get("time")
        if kind == "turn/start":
            turn = int(data.get("turn") or 0)
            current = turn
            order.append(turn)
            turns[turn] = {
                "turn": turn, "start": time, "end": time, "errors": 0, "steps": 0,
                "tool_calls": 0, "folds": 0, "tokens": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
                "human_prompt": "",
            }
            if isinstance(time, int) and isinstance(previous_end, int):
                gap_ms += max(0, time - previous_end)
            continue
        if current is None or current not in turns:
            continue
        record = turns[current]
        if isinstance(time, int):
            record["end"] = time
        if kind == "step/start":
            record["steps"] = int(record["steps"]) + 1
        elif kind == "tool/call":
            record["tool_calls"] = int(record["tool_calls"]) + 1
        elif kind == "assistant/message":
            tokens = record["tokens"]
            assert isinstance(tokens, dict)
            for key, value in usage_of(data).items():
                tokens[key] += value
        elif kind == "compaction/prune":
            record["folds"] = int(record["folds"]) + 1
        elif kind == "user/message":
            source = data.get("source") or {}
            if isinstance(source, dict) and source.get("kind") == "user":
                text = "".join(
                    block.get("text", "") for block in (data.get("content") or [])
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                record["human_prompt"] = text[:60]
        elif kind == "turn/end":
            reason = data.get("reason") or {}
            kind_of_end = reason.get("kind") if isinstance(reason, dict) else None
            record["end_kind"] = kind_of_end
            if kind_of_end == "error":
                record["errors"] = int(record["errors"]) + 1
            previous_end = time if isinstance(time, int) else previous_end

    # The runner's sent turns, so a session turn can be named by the subtask it
    # was asked for. `turns` is cumulative, so sent turn k covers the session
    # turns up to and including it.
    sent: list[dict[str, object]] = []
    boundary = 0
    for entry in report.get("turns") or []:
        upto = int(entry.get("turns") or 0)
        covered = [turn for turn in order if boundary < turn <= upto]
        sent.append({"label": entry.get("label"), "session_turns": covered,
                     "attempts": entry.get("attempts"), "seconds": entry.get("seconds")})
        boundary = upto

    notes = [json.loads(p.read_text()) for p in sorted((NOTES_ROOT / str(report["session_id"]) / "tasks").glob("task-*.json"))]
    note_turns = {}
    for index, note in enumerate(notes, start=1):
        note_turns[f"task-{index:04d}"] = note.get("turn")

    def bucket(turn: int) -> str:
        return "error" if int(turns[turn]["errors"]) > 0 else "completed"

    totals = {
        key: {"tokens": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
              "steps": 0, "tool_calls": 0, "folds": 0, "turns": 0, "ms": 0}
        for key in ("completed", "error")
    }
    for turn in order:
        record = turns[turn]
        target = totals[bucket(turn)]
        target["turns"] = int(target["turns"]) + 1
        target["steps"] = int(target["steps"]) + int(record["steps"])
        target["tool_calls"] = int(target["tool_calls"]) + int(record["tool_calls"])
        target["folds"] = int(target["folds"]) + int(record["folds"])
        if isinstance(record["start"], int) and isinstance(record["end"], int):
            target["ms"] = int(target["ms"]) + (int(record["end"]) - int(record["start"]))
        for key, value in record["tokens"].items():
            target["tokens"][key] += value

    recorder_calls = []
    usage_path = NOTES_ROOT / str(report["session_id"]) / "recorder-usage.jsonl"
    if usage_path.exists():
        recorder_calls = [json.loads(line) for line in usage_path.read_text().splitlines() if line.strip()]
    recorder_error = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    recorder_ok = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    for call in recorder_calls:
        target = recorder_error if bucket(int(call.get("turn") or 0)) == "error" else recorder_ok
        target["calls"] += 1
        target["input_tokens"] += int(call.get("inputTokens") or 0)
        target["output_tokens"] += int(call.get("outputTokens") or 0)
        target["cache_read_tokens"] += int(call.get("cacheReadTokens") or 0)

    notes_on_error = [name for name, turn in note_turns.items() if isinstance(turn, int) and bucket(turn) == "error"]

    return {
        "arm": path.stem,
        "session_id": report["session_id"],
        "pass1": report["pass1"]["total"],
        "pass2": report["pass2"]["total"],
        "session_turns": len(order),
        "sent_turns": len(sent),
        "sent": sent,
        "completed": totals["completed"],
        "error": totals["error"],
        "gap_ms": gap_ms,
        "span_ms": (max(int(turns[t]["end"]) for t in order) - min(int(turns[t]["start"]) for t in order)) if order else 0,
        "recorder_ok": recorder_ok,
        "recorder_error": recorder_error,
        "notes_total": len(notes),
        "notes_on_error_turns": notes_on_error,
        "reported": {
            "steps": report["session"]["steps"],
            "minutes": round(sum(float(t.get("seconds") or 0) for t in report.get("turns") or []) / 60),
            "cost": round(
                (report["session"]["usage"]["input_tokens"] * 1.0
                 + report["session"]["usage"]["cache_read_tokens"] * 0.02
                 + report["session"]["usage"]["output_tokens"] * 4.0) / 1e6, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = []
    header = (f"{'arm':<12}{'p1':>3}{'p2':>4}{'sent':>5}{'sess':>5}{'OK_turn':>8}{'ERR_turn':>9}"
              f"{'OK_min':>7}{'ERR_min':>8}{'gap_min':>8}{'span_min':>9}"
              f"{'OK_¥':>8}{'ERR_¥':>8}{'recOK_¥':>9}{'recERR_¥':>9}{'notes':>6}{'rework':>7}")
    print(header)
    for path in args.paths:
        row = analyse(Path(path))
        rows.append(row)
        completed = row["completed"]
        error = row["error"]
        print(
            f"{row['arm']:<12}{row['pass1']:>3}{row['pass2']:>4}{row['sent_turns']:>5}{row['session_turns']:>5}"
            f"{row['completed']['turns']:>8}{row['error']['turns']:>9}"
            f"{row['completed']['ms'] / 60000:>7.1f}{row['error']['ms'] / 60000:>8.1f}{row['gap_ms'] / 60000:>8.1f}"
            f"{row['span_ms'] / 60000:>9.1f}"
            f"{money(completed['tokens']):>8.3f}{money(error['tokens']):>8.3f}"
            f"{money(row['recorder_ok']):>9.3f}{money(row['recorder_error']):>9.3f}"
            f"{row['notes_total']:>6}{len(row['notes_on_error_turns']):>7}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
