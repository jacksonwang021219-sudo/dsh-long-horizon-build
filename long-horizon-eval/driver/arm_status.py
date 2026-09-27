"""Print one live line per running arm, from each arm's own session log."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import Harness, stats_from_events

for log in sorted(Path("runs").glob("dsh-v*-r*.log")):
    lines = log.read_text().splitlines()
    if not lines or "session " not in lines[0]:
        print(f"{log.stem:<12} (starting)"); continue
    sid = lines[0].split("session ")[1].split(" ")[0]
    turns = len([l for l in lines if "[TURN" in l])
    last = next((l for l in reversed(lines) if "[TURN" in l), "")
    harness = Harness("http://127.0.0.1:1")
    path = harness.log_path(sid)
    if path is None:
        print(f"{log.stem:<12} no log"); continue
    completed = subprocess.run(["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(path)],
                               capture_output=True, text=True, check=True)
    events = [json.loads(l) for l in completed.stdout.splitlines() if l.strip()]
    stats = stats_from_events(events, sid)
    print(f"{log.stem:<12} done_turns={turns:<2} live_turn={stats.turns} steps={stats.steps:<4} "
          f"folds={stats.folds:<2} tokens={stats.usage.total_tokens/1e6:.1f}M | {last[7:70]}")
