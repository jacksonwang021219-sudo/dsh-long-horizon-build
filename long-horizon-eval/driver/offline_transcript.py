"""Read one run's durable transcript offline: no live harness required.

    python3 offline_transcript.py --run-root <dir> [--tail 12] [--grep web_search]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import SESSIONS_ROOT, text_of_content  # noqa: E402

READLOG = Path(__file__).resolve().parent / "readlog.mjs"


def load_log(path: Path) -> list[dict]:
    """Decode one zstd-framed JSONL session artifact."""
    out = subprocess.run(["node", str(READLOG), str(path)], capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--grep", default="")
    parser.add_argument("--all-sessions", action="store_true")
    args = parser.parse_args()
    target = str(Path(args.run_root).resolve())

    matches = []
    for path in SESSIONS_ROOT.glob("*/session-*/session.jsonl.zstd"):
        try:
            events = load_log(path)
        except Exception:  # noqa: BLE001
            continue
        header = next((e for e in events if e.get("type") == "session"), {})
        if str(header.get("cwd") or "") != target:
            continue
        matches.append((len(events), header.get("origin", "root"), path, events))
    matches.sort(reverse=True)
    print(f"该工作区下的会话: {[(n, o, p.parent.name[:22]) for n, o, p, _ in matches]}")
    roots = [m for m in matches if m[1] == "root"] or matches
    for count, origin, path, events in roots[:1 if not args.all_sessions else len(roots)]:
        print(f"\n=== {path.parent.name} ({origin}, {count} events) ===")
        rows = []
        for event in events:
            kind = event.get("type")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if kind == "assistant/message":
                text = text_of_content(data.get("message", {}).get("content"))
                calls = [b.get("name") for b in (data.get("message", {}).get("content") or [])
                         if isinstance(b, dict) and b.get("type") == "tool-call"]
                rows.append(("A", f"[{','.join(c for c in calls if c)}] {text[:300]}"))
            elif kind == "tool/call":
                rows.append(("C", f"{data.get('name')}: {str(data.get('arguments'))[:180]}"))
            elif kind == "tool/result":
                body = text_of_content((data.get("message") or {}).get("content"))
                err = (data.get("error") or {}).get("code")
                rows.append(("R", f"{'ERR ' + str(err) + ' ' if err else ''}{body[:200]}"))
        if args.grep:
            hits = [row for row in rows if args.grep.lower() in row[1].lower()]
            print(f"匹配 {args.grep!r} 的行: {len(hits)}")
            for kind, text in hits[:20]:
                print(f"  {kind} {text[:220]}")
        for kind, text in rows[-args.tail:]:
            print(f"  {kind} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
