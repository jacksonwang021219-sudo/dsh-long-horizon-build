"""Run the clean nine-arm evaluation: standard x3, long-horizon V1 x3, V2 x3.

Every arm gets, before anything starts:

* its own private working root (session cwd) and its own grader root, both under
  the system temp directory and outside the benchmark checkout;
* its own macOS seatbelt profile, applied to its harness process, which denies
  reads and writes of the evaluation workspace and of every other arm's roots;
* its own artifact directory, so no arm can overwrite another's grader output.

Arms run concurrently, one harness process each.

    python3 run_clean_eval.py --out ../runs/clean [--only v1] [--max-subtasks 2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seatbelt import profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT.parent / "deepseek-harness"
SCENARIO = ROOT / "bench" / "AgencyBench" / "AgencyBench-v2" / "Backend" / "scenario1"
PNPM_PATH = "/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback"

ARMS = [
    {"mode": "std", "preset": "standard", "model": f"clean-std-r{index}"} for index in (1, 2, 3)
] + [
    {"mode": "v1", "preset": "lh-v1", "model": f"clean-v1-r{index}"} for index in (1, 2, 3)
] + [
    {"mode": "v2", "preset": "lh-v2", "model": f"clean-v2-r{index}"} for index in (1, 2, 3)
]


def api_key() -> str:
    """The DeepSeek key the harnesses authenticate with."""
    credentials = Path.home() / ".dsh" / ".credentials.yaml"
    text = credentials.read_text()
    for token in text.replace("\n", " ").split():
        if token.startswith("sk-"):
            return token.strip('"\'')
    raise SystemExit("no sk- key found in ~/.dsh/.credentials.yaml")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "clean"))
    parser.add_argument("--only", default="", help="restrict to one mode: std, v1, or v2")
    parser.add_argument("--max-subtasks", type=int, default=5)
    parser.add_argument("--second-pass", action="store_true", default=True)
    parser.add_argument("--turn-timeout", type=float, default=3000.0)
    parser.add_argument("--base-port", type=int, default=63010)
    parser.add_argument("--no-sandbox", action="store_true", help="diagnostics only")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = [arm for arm in ARMS if not args.only or arm["mode"] == args.only]
    for index, arm in enumerate(arms):
        arm["port"] = args.base_port + index
        arm["agent_root"] = Path(tempfile.mkdtemp(prefix="dsh-work-")).resolve()
        arm["grade_root"] = Path(tempfile.mkdtemp(prefix="dsh-grade-")).resolve()

    # Every profile denies the evaluation checkout and the OTHER arms' roots.
    eval_root = str(ROOT)
    for arm in arms:
        others = [str(other["agent_root"]) for other in arms if other is not arm]
        others += [str(other["grade_root"]) for other in arms if other is not arm]
        arm["profile"] = out / f"{arm['model']}.sb"
        arm["profile"].write_text(profile(eval_root, others))

    env = {
        **os.environ,
        "PATH": f"{PNPM_PATH}:{os.environ.get('PATH', '')}",
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DEEPSEEK_API_KEY": api_key(),
    }

    # 1. One harness per arm, each inside its own seatbelt profile.
    harnesses = []
    for arm in arms:
        log = (out / f"harness-{arm['model']}.log").open("w")
        command = ["pnpm", "dsh", "web", "--port", str(arm["port"])]
        if not args.no_sandbox:
            command = ["sandbox-exec", "-f", str(arm["profile"]), *command]
        harnesses.append(subprocess.Popen(command, cwd=HARNESS, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"[CLEAN] harness {arm['model']} port {arm['port']} profile {arm['profile'].name} pid {harnesses[-1].pid}", flush=True)

    # 2. Wait for every harness to answer.
    for arm in arms:
        deadline = time.time() + 180
        while time.time() < deadline:
            probe = subprocess.run(
                ["curl", "-sf", "-o", "/dev/null", f"http://127.0.0.1:{arm['port']}/"],
                capture_output=True,
            )
            if probe.returncode == 0:
                break
            time.sleep(2)
        print(f"[CLEAN] harness {arm['model']} ready={probe.returncode == 0}", flush=True)

    # 3. One runner per arm, in parallel.
    runners = []
    for arm in arms:
        log = (out / f"{arm['model']}.log").open("w")
        command = [
            "python3", "-u", str(Path(__file__).resolve().parent / "bench_agency.py"),
            "--base-url", f"http://127.0.0.1:{arm['port']}",
            "--scenario", str(SCENARIO),
            "--mode", arm["mode"],
            "--preset", arm["preset"],
            "--model-name", arm["model"],
            "--max-subtasks", str(args.max_subtasks),
            "--second-pass",
            "--turn-timeout", str(args.turn_timeout),
            "--turn-retries", "3",
            "--agent-root", str(arm["agent_root"]),
            "--grade-root", str(arm["grade_root"]),
            "--out", str(out / f"{arm['model']}.json"),
        ]
        runners.append(subprocess.Popen(command, cwd=Path(__file__).resolve().parent, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"[CLEAN] runner {arm['model']} pid {runners[-1].pid} -> {out / (arm['model'] + '.log')}", flush=True)

    (out / "arms.json").write_text(json.dumps(
        [{k: str(v) for k, v in arm.items()} for arm in arms], ensure_ascii=False, indent=2,
    ))

    for runner in runners:
        runner.wait()
    for harness in harnesses:
        harness.terminate()
    print("[CLEAN] all runners finished", flush=True)
    for arm in arms:
        report = out / f"{arm['model']}.json"
        if report.exists():
            data = json.loads(report.read_text())
            print(f"[CLEAN] {arm['model']}: pass1={data['pass1']['total']} pass2={data['pass2']['total']} "
                  f"steps={data['session']['steps']} leak={data['leakage']['markers']} "
                  f"cross_arm={data['leakage'].get('cross_arm_calls')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
