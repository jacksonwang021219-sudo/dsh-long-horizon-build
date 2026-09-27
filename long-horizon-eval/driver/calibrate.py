"""Calibrate one scenario's grader on an empty workspace (no agent, no tokens).

Runs the benchmark's own `eval_task.py --eval-only` with a placeholder model
name and reports whether the grader can execute here at all: exit status,
meta_eval.json, and the first grader error lines. Scoring an empty workspace
always yields ~0 — this only proves the scoring pipeline runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TEXT_JUDGE_ENV = """EVAL_TEXT_MODEL=deepseek-flash
EVAL_TEXT_API_BASE_URL=https://api.deepseek.com/v1
EVAL_TEXT_API_KEY={key}
VISION_MODEL=deepseek-flash
EVAL_VISION_PROVIDER=qwen
QWEN_VISION_BASE_URL=https://api.deepseek.com/v1
QWEN_VISION_API_KEY={key}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="path to one scenario directory")
    parser.add_argument("--model-name", default="calibration")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scenario = Path(args.scenario).resolve()
    key = ""
    for line in (Path.home() / ".dsh" / ".credentials.yaml").read_text().splitlines():
        if line.startswith("DEEPSEEK_API_KEY:"):
            key = line.split(":", 1)[1].strip()
    env_file = scenario / f"env-{args.model_name}.env"
    env_file.write_text(
        f"SII_TARGET_MODEL={args.model_name}\n"
        "SII_AGENT_API_KEY=byo-harness-not-used\n"
        "SII_AGENT_API_BASE_URL=https://api.deepseek.com\n"
        "SII_USERNAME=byo\n"
        "SII_PASSWORD=byo\n"
        "SCORER_NAME=rubric\n"
        "MAX_SUBTASK_ATTEMPTS=1\n"
        + (TEXT_JUDGE_ENV.format(key=key) if key else "")
    )
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "eval_task.py", "--eval-only", "--env", env_file.name],
            cwd=scenario, capture_output=True, text=True, timeout=args.timeout,
        )
        status, out, err = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as expired:
        status, out, err = 'timeout', (expired.stdout or b'').decode(errors='replace') if isinstance(expired.stdout, bytes) else (expired.stdout or ''), 'timeout'
    meta = scenario / args.model_name / "meta_eval.json"
    report = {
        "scenario": str(scenario.relative_to(scenario.parents[1])),
        "returncode": status,
        "seconds": round(time.time() - started, 1),
        "meta_written": meta.exists(),
        "scores": [],
        "tail": (out or "")[-1500:],
        "stderr_tail": (err or "")[-800:],
    }
    if meta.exists():
        data = json.loads(meta.read_text())
        report["scores"] = [
            {"name": item.get("name"), "best_score": item.get("best_score")} for item in data.get("subtasks", [])
        ]
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("scenario", "returncode", "seconds", "meta_written")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
