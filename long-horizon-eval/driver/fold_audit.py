"""Classify every fold one arm performed: inside a turn, or across turns.

The window folds for two different reasons, and the session log distinguishes
them by where the shadowed nodes lived: a fold whose shadowed nodes all belong
to the turn that is running at that moment is an **in-turn** fold (the 400K
budget case), while one that spans or targets earlier turns is a **cross-turn**
fold (the retained-window case, which needs more user turns than
`retainUserTurns`).

    python3 fold_audit.py runs/clean/clean-v2-r1.json ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness  # noqa: E402


def decode(session_id: str) -> list[dict[str, object]]:
    """Decode one session's durable log."""
    path = Harness("http://127.0.0.1:1").log_path(session_id)
    if path is None:
        raise SystemExit(f"no durable log for {session_id}")
    completed = subprocess.run(
        ["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
        check=True, capture_output=True, text=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def audit(path: Path) -> dict[str, object]:
    """Split one arm's folds into in-turn and cross-turn."""
    report = json.loads(path.read_text())
    events = decode(str(report["session_id"]))
    turn_of_seq: dict[int, int] = {}
    current = 0
    kinds: Counter[str] = Counter()
    blocks: Counter[str] = Counter()
    detail = []
    for event in events:
        seq = event.get("seq")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("type") == "turn/start":
            current = int(data.get("turn") or current)
            continue
        if isinstance(seq, int):
            turn_of_seq[seq] = current
        if event.get("type") == "compaction/prune":
            shadowed = data.get("shadowedSeqs") or []
            covered = sorted({turn_of_seq.get(int(s), -1) for s in shadowed})
            here = turn_of_seq.get(int(seq), -1)
            kind = "in-turn" if covered == [here] else "cross-turn"
            kinds[kind] += 1
            detail.append({"seq": seq, "kind": kind, "turns": covered, "nodes": len(shadowed),
                           "tokens": data.get("shadowedTokenCount")})
        if event.get("type") == "user/message":
            source = data.get("source") or {}
            if isinstance(source, dict) and source.get("plugin") == "dsh-context-window":
                text = "".join(b.get("text", "") for b in (data.get("content") or []) if isinstance(b, dict))
                for marker in ("<turn-summary", "<folded-history", "<unrecorded-trace"):
                    if marker in text:
                        blocks[marker.lstrip("<")] += 1
    return {
        "arm": path.stem,
        "folds": sum(kinds.values()),
        "in_turn": kinds["in-turn"],
        "cross_turn": kinds["cross-turn"],
        "blocks": dict(blocks),
        "user_turns": len(report.get("turns") or []),
        "session_turns": report["session"]["turns"],
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = []
    print(f"{'arm':<11}{'用户轮':>7}{'会话轮':>7}{'折叠':>5}{'轮内':>6}{'跨轮':>6}   折叠块类型")
    for raw in args.paths:
        row = audit(Path(raw))
        rows.append(row)
        print(f"{row['arm']:<11}{row['user_turns']:>7}{row['session_turns']:>7}{row['folds']:>5}"
              f"{row['in_turn']:>6}{row['cross_turn']:>6}   {row['blocks']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
