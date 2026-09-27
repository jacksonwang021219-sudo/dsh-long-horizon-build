"""Reconstruct files an agent wrote, from its own durable transcript.

Benchmark graders delete the required artifact before each attempt
(`_prepare_result_artifact` / `_prepare_required_artifact`). When a foreign agent
has already produced that artifact, the deletion destroys the deliverable before
it is ever scored. The transcript still holds every write, so this replays the
agent's own file writes back onto disk — no model calls, no guessing.

    python3 recover_artifacts.py --run-root <dir> [--apply]
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


def writes_in(events: list[dict], root: str) -> list[tuple[str, str]]:
    """Every (path, content) the agent wrote under the run root."""
    found: list[tuple[str, str]] = []
    for event in events:
        if event.get("type") != "tool/call":
            continue
        name = event["data"].get("name")
        try:
            args = json.loads(event["data"].get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if name == "write" and isinstance(args.get("file_path"), str):
            found.append((args["file_path"], args.get("content", "")))
        elif name == "bash" and isinstance(args.get("command"), str):
            command = args["command"]
            # A heredoc write is the other way agents create artifacts.
            if "<<'" in command or '<<"' in command:
                found.append((f"<heredoc>{command.splitlines()[0][:80]}", command))
    return [(path, content) for path, content in found if root.split("AgencyBench-v2/")[-1] in path or "<heredoc>" in path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--artifact", default="", help="extra artifact file name to look for")
    args = parser.parse_args()
    target = str(Path(args.run_root).resolve())

    for path in sorted(SESSIONS_ROOT.glob("*/session-*/session.jsonl.zstd")):
        events = decode(path)
        header = next((e for e in events if e.get("type") == "session"), None)
        if header is None or str(header.get("cwd") or "") != target:
            continue
        for written, content in writes_in(events, target):
            if args.artifact and args.artifact not in written and "<heredoc>" not in written:
                continue
            print(f"{path.parent.name[:24]}  ->  {written}  ({len(content)} chars)")
            if args.apply and "<heredoc>" not in written:
                destination = Path(written)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
                print(f"    [WROTE] {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
