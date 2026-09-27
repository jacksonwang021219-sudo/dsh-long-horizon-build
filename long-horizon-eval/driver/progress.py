"""Print one line of live progress for every recent session on the harness."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import Harness, stats_from_events, SESSIONS_ROOT

h = Harness(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:58422")
for item in h.summaries()[:8]:
    sid = item.get("sessionId")
    if not isinstance(sid, str) or item.get("blank"):
        continue
    path = h.log_path(sid)
    if path is None:
        continue
    try:
        events = h.read_log(sid)
    except Exception as error:  # noqa: BLE001
        print(sid, "read error", error)
        continue
    header = next((e for e in events if e.get("type") == "session"), {})
    stats = stats_from_events(events, sid)
    print(f"{sid} running={item.get('running')} cwd={Path(str(header.get('cwd'))).name[:28]} "
          f"turns={stats.turns} steps={stats.steps} records={stats.records} folds={stats.folds} "
          f"total_tokens={stats.usage.total_tokens} out={stats.usage.output_tokens} cache={stats.usage.cache_read_tokens}")
