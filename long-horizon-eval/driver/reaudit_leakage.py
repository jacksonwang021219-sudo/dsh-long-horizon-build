"""Re-run the leak audit on finished runs and rewrite their `leakage` block.

The audit is a pure function of the session log, so a run whose stored audit was
produced by an earlier marker set (or before the temp-dir path normalization)
can be re-measured afterwards. `Score_*` is excluded here for the same reason it
is excluded in `bench_agency.py`: the task prompt names the rubric points, so an
agent writing its own `Score_B` self-check is following the prompt.

    python3 reaudit_leakage.py runs/dsh-v1-r1.json runs/dsh-v2-r1.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_agency import audit_leakage  # noqa: E402
from dsh import Harness  # noqa: E402


def reaudit(path: Path) -> None:
    """Recompute one run's leak audit from its own session log."""
    report = json.loads(path.read_text())
    harness = Harness("http://127.0.0.1:1")
    log = harness.log_path(str(report["session_id"]))
    if log is None:
        raise SystemExit(f"no durable log for {report['session_id']}")
    completed = subprocess.run(
        ["node", str(Path(__file__).resolve().parent / "readlog.mjs"), str(log)],
        check=True, capture_output=True, text=True,
    )
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    root = Path(str(report.get("agent_root") or report.get("run_root") or "/nonexistent"))
    audit = audit_leakage(events, root, Path(str(report["scenario"])))
    report["leakage"] = audit
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{path.stem}: calls={audit['tool_calls']} markers={audit['markers']} "
          f"marker_calls={audit['marker_calls']} outside={audit['paths_outside_root_calls']}")


def main() -> int:
    for argument in sys.argv[1:]:
        reaudit(Path(argument))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
