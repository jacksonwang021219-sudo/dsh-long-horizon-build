"""Compare the notes two arms wrote for their sessions.

Reads each arm's `~/.dsh/notes/<sessionId>/` store and reports, per note, what
the recorder actually produced: the four window fields and their sizes, how much
of each note's conclusions an earlier note already carried (the dedup rule's
measurable effect), and how many recorder attempts it took.

    python3 compare_notes.py --run ../runs/dsh-v1-r1.json --run ../runs/dsh-v2-r1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NOTES_ROOT = Path.home() / ".dsh" / "notes"
FIELDS = ("taskDefinition", "strategyAndProgress", "keyConclusions", "deliveryAndAcceptance")

# What a note should never mention if the agent never saw the grading side. The
# task prompt itself says `Score_*` and "rubric", so neither is listed: only the
# grader's own file names and its scored artifacts are.
LEAK_TERMS = ("evalspace", "eval_task", "meta_eval", "failed_points", "check_rubric")

# Word shingles are the dedup unit: a repeated conclusion usually returns with
# different punctuation, so exact-string comparison would miss it.
SHINGLE = 8


def shingles(text: str) -> set[str]:
    """Word n-grams of one field, lowercased and stripped of punctuation."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < SHINGLE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index:index + SHINGLE]) for index in range(len(words) - SHINGLE + 1)}


def load_session(session_id: str) -> dict[str, object]:
    """Read one session's notes, its recorder log, and its usage log."""
    directory = NOTES_ROOT / session_id
    notes = []
    for path in sorted((directory / "tasks").glob("task-*.json")):
        notes.append(json.loads(path.read_text()))
    log = (directory / "recorder.log").read_text().splitlines() if (directory / "recorder.log").exists() else []
    usage = []
    usage_path = directory / "recorder-usage.jsonl"
    if usage_path.exists():
        usage = [json.loads(line) for line in usage_path.read_text().splitlines() if line.strip()]
    index = json.loads((directory / "index.json").read_text()) if (directory / "index.json").exists() else {}
    return {"session_id": session_id, "notes": notes, "log": log, "usage": usage, "index": index}


def analyse(label: str, session_id: str) -> dict[str, object]:
    """Reduce one session's notes to the comparison's numbers."""
    data = load_session(session_id)
    notes = data["notes"]
    seen: set[str] = set()
    rows = []
    for note in notes:
        summary = note.get("windowSummary") or {}
        carried = 0.0
        own = shingles(summary.get("keyConclusions", ""))
        if own and seen:
            carried = len(own & seen) / len(own)
        seen |= own
        rows.append({
            "id": note.get("id"),
            "turn": note.get("turn"),
            "range": f"{note.get('start')}-{note.get('end')}",
            "title": (summary.get("taskDefinition") or "")[:60],
            "chars": {field: len(summary.get(field, "")) for field in FIELDS},
            "total": sum(len(summary.get(field, "")) for field in FIELDS),
            "empty_fields": [field for field in FIELDS if not summary.get(field, "").strip()],
            "conclusions_carried_over": round(carried, 3),
            "experiences": len(note.get("reusableExperience") or []),
        })
    views = re.findall(r"(\d+) window message\(s\)", "\n".join(data["log"]))
    # Two notes covering overlapping log ranges mean the same work was recorded
    # twice — the in-turn path can do that when a turn folds more than once.
    ranges = [(int(note.get("start", 0)), int(note.get("end", 0))) for note in notes]
    overlaps = sum(
        1 for index, (start, end) in enumerate(ranges)
        if any(start <= other_end and other_start <= end for other_start, other_end in ranges[:index])
    )
    # Leaked grader knowledge, as it would persist: a term a note uses that only
    # the grading side of the scenario defines.
    note_text = "\n".join(
        summary
        for note in notes
        for field in FIELDS
        for summary in [(note.get("windowSummary") or {}).get(field, "")]
    ).lower()
    leak_terms = {term: note_text.count(term) for term in LEAK_TERMS if term in note_text}
    return {
        "leak_terms_in_notes": leak_terms,
        "label": label,
        "session_id": session_id,
        "notes": len(notes),
        "rows": rows,
        "recorder_log_lines": len(data["log"]),
        "recorder_failures": len([line for line in data["log"] if "not accepted" in line or "GAVE UP" in line or "threw" in line]),
        "window_messages": views,
        "usage_calls": len(data["usage"]),
        "usage_accepted": len([row for row in data["usage"] if row.get("accepted")]),
        "index_conclusions_present": sum(1 for task in (data["index"].get("tasks") or []) if task.get("conclusions")),
        "topics": len(data["index"].get("topics") or []),
        "types": len(data["index"].get("types") or []),
        "mean_total_chars": round(sum(row["total"] for row in rows) / len(rows), 1) if rows else 0,
        "mean_carried": round(sum(row["conclusions_carried_over"] for row in rows) / len(rows), 3) if rows else 0,
        "overlapping_note_ranges": overlaps,
        "chars_min": min((row["total"] for row in rows), default=0),
        "chars_max": max((row["total"] for row in rows), default=0),
        "notes_over_2000_chars": len([row for row in rows if row["total"] > 2000]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="path to a run report JSON")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    results = []
    for path in args.run:
        report = json.loads(Path(path).read_text())
        label = Path(path).stem
        results.append(analyse(label, str(report["session_id"])))
        print(f"== {label}  session {report['session_id']}")
        for row in results[-1]["rows"]:
            print(
                f"   {row['id']} turn {row['turn']} log {row['range']} chars={row['total']:>5} "
                f"carried={row['conclusions_carried_over']:>5} empty={','.join(row['empty_fields']) or '-'} "
                f"| {row['title']}"
            )
        item = results[-1]
        print(
            f"   notes={item['notes']} mean_chars={item['mean_total_chars']} mean_carried={item['mean_carried']} "
            f"recorder_calls={item['usage_calls']} accepted={item['usage_accepted']} failures={item['recorder_failures']} "
            f"window_messages={item['window_messages']} index_conclusions={item['index_conclusions_present']} "
            f"topics={item['topics']} types={item['types']} leak_terms={item['leak_terms_in_notes']}\n"
            f"   chars min/max={item['chars_min']}/{item['chars_max']} over-2000={item['notes_over_2000_chars']}/{item['notes']} "
            f"overlapping_ranges={item['overlapping_note_ranges']}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
