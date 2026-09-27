"""Drive a running DeepSeek Harness Web instance over its local HTTP RPC API.

The harness exposes the same RPC surface its own GUI uses: a POST to
``/api/<method>`` with a ``client-request`` envelope. Sessions are created with
an explicit agent preset, so one instance can run the same task under two
compositions, and each session's durable JSONL log is the evidence for both
scoring and token accounting.

Module is import-only; ``python3 -m`` scripts in this directory use it.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
READLOG = HERE / "readlog.mjs"
SESSIONS_ROOT = Path.home() / ".dsh" / "sessions"


class HarnessError(RuntimeError):
    """An RPC or log operation the driver could not complete."""


@dataclass
class Usage:
    """Token accounting accumulated over one session's model calls."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Every token the provider accounted for, cached reads included."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.reasoning_tokens
        )

    def as_dict(self) -> dict[str, int]:
        """Report the accounting as plain JSON."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class SessionStats:
    """Durable facts read back from one session log."""

    session_id: str
    turns: int = 0
    steps: int = 0
    tool_calls: int = 0
    records: int = 0
    folds: int = 0
    turns_without_record: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    last_assistant_text: str = ""

    def as_dict(self) -> dict[str, object]:
        """Report the statistics as plain JSON."""
        return {
            "session_id": self.session_id,
            "turns": self.turns,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "records": self.records,
            "folds": self.folds,
            "turns_without_record": self.turns_without_record,
            "usage": self.usage.as_dict(),
            "last_assistant_text": self.last_assistant_text,
        }


class Harness:
    """One running ``dsh web`` instance, addressed by base URL."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._rpc_id = 0

    def rpc(self, method: str, payload: dict[str, object], attempts: int = 5) -> object:
        """Call one client-request RPC method and return its value.

        A benchmark run outlives transient transport failures — a harness
        restart, a busy socket — so the call retries with backoff instead of
        abandoning a whole scenario run.
        """
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._rpc_once(method, payload)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
                last = error
                if attempt < attempts:
                    time.sleep(min(30.0, 3.0 * attempt))
        raise HarnessError(f"{method} unreachable after {attempts} attempts: {last}")

    def _rpc_once(self, method: str, payload: dict[str, object]) -> object:
        """One client-request round trip."""
        self._rpc_id += 1
        body = json.dumps(
            {
                "type": "client-request",
                "rpcId": str(self._rpc_id),
                "method": method,
                "payload": payload,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/{method}",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        result = decoded.get("result", {})
        if not result.get("ok"):
            raise HarnessError(f"{method} failed: {json.dumps(result)[:400]}")
        return result.get("value")

    def presets(self) -> list[dict[str, object]]:
        """Every preset the roster currently supplies."""
        value = self.rpc("agentPreset.list", {})
        assert isinstance(value, dict)
        presets = value.get("presets", [])
        assert isinstance(presets, list)
        return presets

    def create_session(self, cwd: str, preset: str) -> str:
        """Create one session on an explicit preset and return its id."""
        value = self.rpc("session.create", {"cwd": cwd, "agentPreset": preset})
        assert isinstance(value, dict)
        session_id = value.get("sessionId") or value.get("id")
        if not isinstance(session_id, str):
            raise HarnessError(f"session.create returned no session id: {json.dumps(value)[:300]}")
        return session_id

    def prompt(self, session_id: str, text: str, mode: str = "queue") -> None:
        """Send one user turn into an existing session."""
        self.rpc("session.prompt", {
            "sessionId": session_id,
            "mode": mode,
            "content": [{"type": "text", "text": text}],
        })

    def cancel(self, session_id: str) -> None:
        """Abort the session's active turn."""
        self.rpc("session.cancel", {"sessionId": session_id})

    def summaries(self) -> list[dict[str, object]]:
        """Every session summary the host currently reports."""
        value = self.rpc("session.list", {})
        assert isinstance(value, dict)
        items = value.get("items", [])
        assert isinstance(items, list)
        return items

    def running(self, session_id: str) -> bool:
        """Whether the named session currently has a live turn."""
        for item in self.summaries():
            if item.get("sessionId") == session_id:
                return bool(item.get("running"))
        return False

    def wait_idle(self, session_id: str, timeout: float = 3600.0, poll: float = 5.0) -> bool:
        """Wait until the session stops running; returns False on timeout."""
        deadline = time.time() + timeout
        # Give the turn a moment to start before trusting a not-running reading.
        time.sleep(min(3.0, poll))
        while time.time() < deadline:
            if not self.running(session_id):
                return True
            time.sleep(poll)
        return False

    def log_path(self, session_id: str) -> Path | None:
        """Locate the durable JSONL artifact of one session."""
        matches = list(SESSIONS_ROOT.glob(f"*/session-{session_id}/session.jsonl.zstd"))
        if matches:
            return matches[0]
        matches = list(SESSIONS_ROOT.glob(f"*/{session_id}/session.jsonl.zstd"))
        return matches[0] if matches else None

    def read_log(self, session_id: str) -> list[dict[str, object]]:
        """Decode one session's durable log into event dictionaries."""
        path = self.log_path(session_id)
        if path is None:
            raise HarnessError(f"no durable log found for session {session_id}")
        completed = subprocess.run(
            ["node", str(READLOG), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        events = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events


def text_of_content(content: object) -> str:
    """Join the text blocks of one message content array."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()


def child_session_ids(session_id: str) -> list[str]:
    """Every session whose header names this one as its parent, recursively.

    A forked note keeper runs in its own session, so its tokens are NOT in the
    parent's log. Comparing total cost without them would undercount the
    long-horizon arm.
    """
    found: list[str] = []
    for path in SESSIONS_ROOT.rglob('session.jsonl.zstd'):
        try:
            completed = subprocess.run(
                ['node', str(READLOG), str(path)], check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            continue
        for line in completed.stdout.splitlines():
            if '"type":"session"' not in line:
                continue
            try:
                header = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent = header.get('parentSession')
            own = header.get('id')
            if isinstance(parent, str) and isinstance(own, str) and parent == session_id:
                found.append(own)
            break
    children: list[str] = []
    for child in found:
        children.append(child)
        children.extend(child_session_ids(child))
    return children


def stats_from_events(events: list[dict[str, object]], session_id: str) -> SessionStats:
    """Fold one decoded log into the statistics the comparison reports."""
    stats = SessionStats(session_id=session_id)
    recorded_turns: set[int] = set()
    turns_with_work: dict[int, bool] = {}
    for event in events:
        kind = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if kind == "turn/end":
            turn = data.get("turn")
            if isinstance(turn, int):
                stats.turns = max(stats.turns, turn)
        elif kind == "step/start":
            stats.steps += 1
        elif kind == "tool/call":
            stats.tool_calls += 1
        elif kind == "assistant/message":
            turn = data.get("turn")
            if isinstance(turn, int):
                turns_with_work[turn] = True
            usage = data.get("usage")
            if isinstance(usage, dict):
                stats.usage.calls += 1
                for key, field_name in (
                    ("inputTokens", "input_tokens"),
                    ("outputTokens", "output_tokens"),
                    ("cacheReadTokens", "cache_read_tokens"),
                    ("cacheWriteTokens", "cache_write_tokens"),
                    ("reasoningTokens", "reasoning_tokens"),
                ):
                    value = usage.get(key)
                    if isinstance(value, int):
                        setattr(stats.usage, field_name, getattr(stats.usage, field_name) + value)
            message = data.get("message")
            if isinstance(message, dict):
                text = text_of_content(message.get("content"))
                if text:
                    stats.last_assistant_text = text
        elif kind == "tool/result":
            meta = data.get("meta")
            if isinstance(meta, dict) and meta.get("kind") == "dsh-context-window/record":
                stats.records += 1
                turn = meta.get("turn")
                if isinstance(turn, int):
                    recorded_turns.add(turn)
        elif kind == "compaction/prune":
            stats.folds += 1
    stats.turns_without_record = sorted(
        turn for turn in turns_with_work if turn not in recorded_turns
    )
    return stats
