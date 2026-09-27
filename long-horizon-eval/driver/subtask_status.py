"""Per-subtask, per-arm status: did the turn run cleanly, and what did it score?

A turn that needed a resend did not run to completion the first time, so its
subtask's score measures "the model plus the interruption", not the model. This
splits every subtask into `clean` (one attempt, no transport error) and
`interrupted`, so the arms can be compared on the runs that actually happened.

    python3 subtask_status.py runs/dsh-*-r*.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness, stats_from_events  # noqa: E402


def turn_errors(session_id: str) -> set[int]:
    """Session-turn numbers whose `turn/end` carried an error reason."""
    path = Harness("http://127.0.0.1:1").log_path(session_id)
    if path is None:
        raise SystemExit(f"no durable log for {session_id}")
    completed = subprocess.run(
        ["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
        check=True, capture_output=True, text=True,
    )
    errored: set[int] = set()
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "turn/end":
            continue
        data = event.get("data") or {}
        reason = data.get("reason") or {}
        if isinstance(reason, dict) and reason.get("kind") == "error":
            errored.add(int(data.get("turn") or 0))
    return errored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report_rows = []
    for raw in args.paths:
        path = Path(raw)
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or "session_id" not in report:
            continue  # not a run report (arms.json, metrics dumps)
        errored = turn_errors(str(report["session_id"]))
        boundary = 0
        rows = []
        for entry in report.get("turns") or []:
            upto = int(entry.get("turns") or 0)
            covered = {turn for turn in range(boundary + 1, upto + 1)}
            boundary = upto
            rows.append({
                "label": entry.get("label"),
                "attempts": entry.get("attempts"),
                "errored_turns": sorted(covered & errored),
            })
        p1 = {item["subtask"]: item["best_score"] for item in report["pass1"]["subtasks"]}
        p2 = {item["subtask"]: item["best_score"] for item in report["pass2"]["subtasks"]}
        subtasks = []
        for index in range(1, 6):
            name = f"subtask{index}"
            first = next((row for row in rows if row["label"] == name), None)
            retry = next((row for row in rows if row["label"] == f"{name}-retry"), None)
            subtasks.append({
                "subtask": name,
                "pass1_attempts": first["attempts"] if first else None,
                "pass1_clean": bool(first and first["attempts"] == 1),
                "retry_attempts": retry["attempts"] if retry else 0,
                "retry_clean": bool(retry and retry["attempts"] == 1),
                "pass1": p1.get(name),
                "pass2": p2.get(name),
            })
        report_rows.append({"arm": path.stem, "subtasks": subtasks})

    # Cross-arm view: every clean pass-1 measurement, by subtask.
    print(f"{'arm':<12}" + "".join(f"{f's{i} clr/p1':>11}" for i in range(1, 6)))
    for row in report_rows:
        cells = []
        for item in row["subtasks"]:
            mark = "clean" if item["pass1_clean"] else f"x{item['pass1_attempts']}"
            cells.append(f"{mark}/{item['pass1']}")
        print(f"{row['arm']:<12}" + "".join(f"{cell:>11}" for cell in cells))

    print()
    print(f"{'subtask':<10}{'V1 clean (n)':>16}{'V1 mean':>9}{'V2 clean (n)':>16}{'V2 mean':>9}")
    for index in range(1, 6):
        name = f"subtask{index}"
        cells = {}
        for arm_prefix in ("clean-v1", "clean-v2"):
            scores = [
                item["pass1"] for row in report_rows
                if row["arm"].startswith(arm_prefix)
                for item in row["subtasks"]
                if item["subtask"] == name and item["pass1_clean"]
            ]
            cells[arm_prefix] = scores
        v1, v2 = cells["clean-v1"], cells["clean-v2"]
        mean = lambda values: f"{sum(values) / len(values):.1f}" if values else "—"  # noqa: E731
        print(f"{name:<10}{str(v1) + f' ({len(v1)})':>16}{mean(v1):>9}"
              f"{str(v2) + f' ({len(v2)})':>16}{mean(v2):>9}")

    print()
    for row in report_rows:
        clean = [item for item in row["subtasks"] if item["pass1_clean"]]
        retried = [item for item in row["subtasks"] if item["retry_attempts"]]
        print(f"{row['arm']:<12} clean pass-1 subtasks {len(clean)}/5 "
              f"({[item['subtask'][-1] + ':' + str(item['pass1']) for item in clean]}) "
              f"| retries {[(item['subtask'][-1], item['retry_attempts'], item['retry_clean'], item['pass1'], item['pass2']) for item in retried]}")

    if args.out:
        Path(args.out).write_text(json.dumps(report_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
