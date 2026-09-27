"""Total token accounting for one run, including its forked recorder sessions."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh import Harness, child_session_ids, stats_from_events

def total(base_url: str, session_id: str) -> dict:
    harness = Harness(base_url)
    own = stats_from_events(harness.read_log(session_id), session_id)
    children = []
    for child in child_session_ids(session_id):
        try:
            stats = stats_from_events(harness.read_log(child), child)
        except Exception:
            continue
        children.append(stats)
    def add(field: str) -> int:
        return getattr(own.usage, field) + sum(getattr(c.usage, field) for c in children)
    return {
        "session": session_id,
        "own_total": own.usage.total_tokens,
        "child_sessions": len(children),
        "child_total": sum(c.usage.total_tokens for c in children),
        "calls": own.usage.calls + sum(c.usage.calls for c in children),
        "input_tokens": add("input_tokens"),
        "cache_read_tokens": add("cache_read_tokens"),
        "output_tokens": add("output_tokens"),
        "reasoning_tokens": add("reasoning_tokens"),
        "total_tokens": add("total_tokens"),
        "steps": own.steps,
        "folds": own.folds,
    }

if __name__ == "__main__":
    base = sys.argv[1]
    for session_id in sys.argv[2:]:
        print(json.dumps(total(base, session_id), ensure_ascii=False, indent=2))
