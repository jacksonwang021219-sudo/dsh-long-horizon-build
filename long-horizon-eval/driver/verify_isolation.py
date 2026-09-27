"""Prove, per arm, that nothing reached the grading side or another arm.

The in-run audit matches patterns, so a typo of the arm's own root (`dsh-work-idqrpnvpk`
for `dsh-work-idqrnvpk`) looks like a cross-arm reach. This re-checks the same
logs against the *actual* roots from `runs/clean/arms.json`, plus the evaluation
checkout path and the durable session store, and separates "named a real other
arm" from "mistyped its own path".

    python3 verify_isolation.py --runs runs/clean --eval-root .
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness  # noqa: E402


def tool_arguments(session_id: str) -> list[tuple[str, str]]:
    """Every tool call of one session, as `(tool name, arguments)`."""
    path = Harness("http://127.0.0.1:1").log_path(session_id)
    if path is None:
        raise SystemExit(f"no durable log for {session_id}")
    completed = subprocess.run(
        ["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
        check=True, capture_output=True, text=True,
    )
    calls: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "tool/call":
            continue
        data = event.get("data") or {}
        arguments = data.get("arguments")
        if isinstance(arguments, str):
            calls.append((str(data.get("name")), arguments))
    return calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/clean")
    parser.add_argument("--eval-root", default=".")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    runs = Path(args.runs)
    eval_root = str(Path(args.eval_root).resolve())
    arms = json.loads((runs / "arms.json").read_text())
    roots = {arm["model"]: (arm["agent_root"], arm["grade_root"]) for arm in arms}

    rows = []
    print(f"{'arm':<14}{'calls':>7}{'other_arm_root':>16}{'own_root_typo':>15}{'eval_checkout':>15}{'session_store':>15}{'total_hits':>12}")
    for arm in arms:
        model = arm["model"]
        report_path = runs / f"{model}.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        own_agent, own_grade = roots[model]
        others = [root for name, pair in roots.items() if name != model for root in pair]
        calls = tool_arguments(str(report["session_id"]))
        other_hits = typo_hits = eval_hits = store_hits = 0
        for _, arguments in calls:
            if any(root in arguments for root in others):
                other_hits += 1
            elif "dsh-work-" in arguments and own_agent not in arguments and own_agent[10:20] in arguments:
                typo_hits += 1
            if eval_root in arguments:
                eval_hits += 1
            if "/.dsh/sessions/" in arguments:
                store_hits += 1
        rows.append({
            "arm": model, "calls": len(calls), "other_arm_root": other_hits,
            "own_root_typo": typo_hits, "eval_checkout": eval_hits, "session_store": store_hits,
        })
        print(f"{model:<14}{len(calls):>7}{other_hits:>16}{typo_hits:>15}{eval_hits:>15}{store_hits:>15}"
              f"{other_hits + eval_hits + store_hits:>12}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    reached = sum(row["other_arm_root"] + row["eval_checkout"] + row["session_store"] for row in rows)
    print(f"\ntotal real reaches into another arm or the grading side: {reached}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
