"""Check whether the notes' concrete claims appear in the session they describe.

A note is a summary, so most of its prose cannot be checked mechanically. Its
*concrete* parts can: every path and every command a note names either occurred
in the session it claims to record, or it was invented. This walks each note's
four fields, extracts the path-like tokens, and reports how many of them appear
in that session's own log — plus the unmatched ones for a human to read.

    python3 verify_notes.py --run ../runs/dsh-v1-r1.json --run ../runs/dsh-v2-r1.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness  # noqa: E402

NOTES_ROOT = Path.home() / ".dsh" / "notes"
FIELDS = ("taskDefinition", "strategyAndProgress", "keyConclusions", "deliveryAndAcceptance")

# Path-like tokens: at least one slash and a recognisable file or directory.
PATH_TOKEN = re.compile(r"(?<![\w-])(?:~|\.{1,2}|/)?(?:[\w.@+-]+/)+[\w.@+-]*\.?[\w@+-]*")
# Tokens too generic to mean anything as evidence.
IGNORED = {"and/or", "either/or", "i/o", "he/she", "n/a", "24/7", "std::thread"}
# File-extension or rooted paths: slash-separated word lists (`push/tryPop/wait`)
# also match PATH_TOKEN but name no file, so they are reported separately.
CONCRETE = re.compile(r"^(?:~|/|\.{1,2}/)|/[\w@+-]+\.[A-Za-z0-9]{1,6}(?:/|$)")


def is_concrete(token: str) -> bool:
    """Whether a token names a file system location rather than a slash list."""
    return CONCRETE.search(token) is not None


def log_text(session_id: str) -> str:
    """The session's raw log, decoded, as one searchable string."""
    harness = Harness("http://127.0.0.1:1")
    path = harness.log_path(session_id)
    if path is None:
        raise SystemExit(f"no durable log for {session_id}")
    completed = subprocess.run(["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
                               check=True, capture_output=True, text=True)
    return completed.stdout


def check(label: str, session_id: str) -> dict[str, object]:
    """Report the grounded fraction of one session's note claims."""
    directory = NOTES_ROOT / session_id / "tasks"
    text = log_text(session_id)
    rows = []
    total = grounded = 0
    for path in sorted(directory.glob("task-*.json")):
        note = json.loads(path.read_text())
        summary = note.get("windowSummary") or {}
        tokens: list[str] = []
        for field in FIELDS:
            tokens.extend(PATH_TOKEN.findall(summary.get(field, "")))
        tokens = [token for token in dict.fromkeys(tokens) if token.lower() not in IGNORED and len(token) > 3]
        missing = [token for token in tokens if token not in text]
        total += len(tokens)
        grounded += len(tokens) - len(missing)
        rows.append({
            "id": note.get("id"),
            "tokens": len(tokens),
            "missing": missing,
            "missing_concrete": [token for token in missing if is_concrete(token)],
        })
    return {
        "label": label,
        "session_id": session_id,
        "tokens": total,
        "grounded": grounded,
        "grounded_share": round(grounded / total, 3) if total else None,
        "missing_concrete": sum(len(row["missing_concrete"]) for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    results = []
    for path in args.run:
        report = json.loads(Path(path).read_text())
        result = check(Path(path).stem, str(report["session_id"]))
        results.append(result)
        print(f"== {result['label']}: {result['grounded']}/{result['tokens']} path tokens "
              f"({result['grounded_share']}) appear in the session log; "
              f"{result['missing_concrete']} unmatched token(s) name a concrete location")
        for row in result["rows"]:
            print(f"   {row['id']}: {row['tokens'] - len(row['missing'])}/{row['tokens']} grounded")
            for token in row["missing_concrete"][:8]:
                print(f"      unmatched path: {token}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
