"""Restore the deliverable a grader deleted, from the agent's own transcript.

The benchmark's `_prepare_*artifact*` step unlinks the required artifact before
each attempt. For a foreign agent that already finished, that destroys the work
before it is scored. Every write the agent performed is in its durable log, so
the last write to the required path is replayed byte-for-byte.

    python3 restore_from_transcript.py --run-root <dir> --artifact <name> [--apply]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import SESSIONS_ROOT  # noqa: E402

READLOG = Path(__file__).resolve().parent / "readlog.mjs"


def decode(path: Path) -> list[dict]:
    """Decode one session artifact, tolerating a torn final line."""
    out = subprocess.run(["node", str(READLOG), str(path)], capture_output=True, text=True).stdout
    events = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--artifact", required=True, help="file name, e.g. web_info.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()

    latest: tuple[str, str] | None = None
    for path in sorted(SESSIONS_ROOT.glob("*/session-*/session.jsonl.zstd")):
        events = decode(path)
        header = next((e for e in events if e.get("type") == "session"), None)
        if header is None or str(header.get("cwd") or "") != str(root):
            continue
        if len(events) < 20:
            continue
        for event in events:
            if event.get("type") != "tool/call" or event["data"].get("name") != "write":
                continue
            try:
                written = json.loads(event["data"].get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            target = str(written.get("file_path") or "")
            if not target.endswith(args.artifact):
                continue
            resolved = Path(target) if target.startswith("/") else root / target
            latest = (str(resolved), written.get("content") or "")
    if latest is None:
        print(f"[MISS] no write of {args.artifact} in the transcript of {root}")
        return 1
    destination, content = latest
    print(f"[FOUND] {destination} ({len(content)} chars, from the transcript)")
    if args.apply:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(content)
        print(f"[WROTE] {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
