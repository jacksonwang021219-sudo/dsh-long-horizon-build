"""Run AgencyBench option C: nine scenarios, both modes, bounded concurrency.

Each scenario keeps the benchmark's own grader and its own workspace layout:

* `subtasks` layout — one `subtask<N>/workspace/attempt_01` per subtask, seeded
  from the previous subtask's output, graded by `eval_task.py --eval-only`;
* `flat` layout — one `workspace` for the whole scenario;
* `shim` grader — the scenario ships no `--eval-only`, so the unmodified grader
  is driven through `eval_only_shim.py`;
* `capture` — the scenario's runner composes its own prompt (result-file paths,
  loss thresholds), so the prompts are captured from the grader itself.

    python3 run_optionc.py --base-url http://127.0.0.1:PORT [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = ROOT / "bench" / "AgencyBench" / "AgencyBench-v2"
OUT = ROOT / "runs" / "optionC"

# (slug, scenario path relative to AgencyBench-v2, runner options)
SCENARIOS: list[tuple[str, str, dict[str, object]]] = [
    ("be1", "Backend/scenario1", {"layout": "subtasks", "grader": "eval-only", "second_pass": True}),
    ("be3", "Backend/scenario3", {"layout": "subtasks", "grader": "shim", "second_pass": True}),
    ("co1", "Code/scenario1", {"layout": "subtasks", "grader": "shim", "second_pass": True}),
    ("co6", "Code/scenario6", {"layout": "flat", "grader": "shim", "seed_template": True,
                               "capture": True, "second_pass": False}),
    ("co7", "Code/scenario7", {"layout": "flat", "grader": "shim", "seed_template": True,
                               "capture": True, "second_pass": False}),
    ("re2", "Research/scenario2", {"layout": "flat", "grader": "shim", "second_pass": False}),
    ("re3", "Research/scenario3", {"layout": "flat", "grader": "shim", "second_pass": False}),
    ("re4", "Research/scenario4", {"layout": "flat", "grader": "shim", "capture": True, "second_pass": False}),
    ("re5", "Research/scenario5", {"layout": "flat", "grader": "shim", "capture": True, "second_pass": False}),
]

MODES: list[tuple[str, str]] = [("standard", "standard"), ("long-horizon", "long-horizon")]


def run_one(base_url: str, slug: str, scenario: str, options: dict[str, object],
            mode: str, preset: str, turn_timeout: float) -> dict[str, object]:
    """Run one scenario under one preset and report the outcome."""
    model_name = f"{'std' if preset == 'standard' else 'lh'}-{slug}"
    command = [
        sys.executable, "-u", str(HERE / "bench_agency.py"),
        "--base-url", base_url,
        "--scenario", str(BENCH / scenario),
        "--mode", mode,
        "--preset", preset,
        "--model-name", model_name,
        "--max-subtasks", "5",
        "--turn-timeout", str(turn_timeout),
        "--layout", str(options["layout"]),
        "--grader", str(options["grader"]),
        "--out", str(OUT / f"{model_name}.json"),
    ]
    if options.get("second_pass"):
        command.append("--second-pass")
    if options.get("seed_template"):
        command.append("--seed-template")
    if options.get("capture"):
        command.append("--prompts-capture")
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{model_name}.log"
    started = time.time()
    with log.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=HERE, text=True)
    return {
        "model_name": model_name,
        "scenario": scenario,
        "preset": preset,
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "out": str(OUT / f"{model_name}.json"),
        "log": str(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--turn-timeout", type=float, default=2400.0)
    parser.add_argument("--only", default="", help="comma-separated scenario slugs to run")
    args = parser.parse_args()

    wanted = {slug for slug in args.only.split(",") if slug}
    jobs = [
        (slug, scenario, options, mode, preset)
        for slug, scenario, options in SCENARIOS
        for mode, preset in MODES
        if not wanted or slug in wanted
    ]
    print(f"[PLAN] {len(jobs)} runs, {args.workers} concurrent", flush=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, args.base_url, slug, scenario, options, mode, preset, args.turn_timeout):
                (slug, mode)
            for slug, scenario, options, mode, preset in jobs
        }
        for future in futures:
            slug, mode = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                result = {"model_name": f"{slug}-{mode}", "error": repr(error)}
            results.append(result)
            print(f"[DONE] {result.get('model_name')} rc={result.get('returncode')} "
                  f"{result.get('seconds')}s", flush=True)
    (OUT / "batch-summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
