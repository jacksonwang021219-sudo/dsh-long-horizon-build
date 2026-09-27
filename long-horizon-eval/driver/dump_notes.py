"""Dump two arms' notes side by side, turn by turn, for a human read.

    python3 dump_notes.py runs/dsh-v1-r1.json runs/dsh-v2-r1.json > /tmp/notes.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTES_ROOT = Path.home() / ".dsh" / "notes"
FIELDS = ("taskDefinition", "strategyAndProgress", "keyConclusions", "deliveryAndAcceptance")


def notes_of(session_id: str) -> list[dict[str, object]]:
    """Every task note of one session, oldest first."""
    directory = NOTES_ROOT / session_id / "tasks"
    return [json.loads(path.read_text()) for path in sorted(directory.glob("task-*.json"))]


def main() -> int:
    runs = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
    arms = [(Path(path).stem, notes_of(str(report["session_id"]))) for path, report in zip(sys.argv[1:], runs)]
    turns = max((len(notes) for _, notes in arms), default=0)
    for index in range(turns):
        print("=" * 100)
        print(f"TURN {index + 1}")
        for label, notes in arms:
            print("-" * 100)
            if index >= len(notes):
                print(f"[{label}] no note")
                continue
            note = notes[index]
            summary = note.get("windowSummary") or {}
            print(f"[{label}] {note.get('id')} log {note.get('start')}-{note.get('end')} "
                  f"({sum(len(summary.get(field, '')) for field in FIELDS)} chars)")
            for field in FIELDS:
                print(f"  <{field}> {summary.get(field, '')}")
            if note.get("reusableExperience"):
                print(f"  <reusable_experience> {json.dumps(note['reusableExperience'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
