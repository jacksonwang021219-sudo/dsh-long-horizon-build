"""Run one AgencyBench scenario under two agent presets and score both.

The benchmark's own rollout path needs `sii_agent_sdk` plus a Node bridge that
is not distributed, so this runner drives the tasks with a bring-your-own agent
(DeepSeek Harness over its local RPC API) and uses the benchmark's unmodified
grader (`eval_task.py --eval-only`) on the deliverables it produces.

Protocol per mode, mirroring the benchmark's own flow:

* subtasks run in order, each in its own ``subtask<N>/workspace/attempt_01``
  directory, seeded from the previous subtask's output;
* every subtask is one user turn, so the session accumulates the multi-turn
  history the context policy is meant to manage;
* after the first pass the grader scores every subtask, and each failing
  subtask gets one more user turn carrying the rubric feedback (the benchmark's
  own attempt-2 / Pass@2 loop);
* the grader scores the final state.

ISOLATION. The agent must not be able to read the grader. The benchmark's own
rollout runs the agent inside a sandbox that mounts only the workspace; a
bring-your-own agent has no such sandbox, and the naive layout — session cwd
inside the scenario directory — puts `eval_task.py` (which holds the whole
rubric, the scoring commands and their expected values) one `..` away, next to
the previous arms' workspaces and their scored `meta_eval.json`. So:

* the session's cwd is a private, randomly named root holding nothing but the
  arm's own workspaces;
* the grader runs out of a second, unrelated private root, pointed at a copy of
  `description.json` through ``--description`` (the grader derives its whole
  run tree from that path's parent);
* the deliverables are synced into the grader's root only after the turns are
  over.

``audit_leakage`` then measures what the agent actually touched, so a run that
still reaches for the rubric is visible in the report instead of being taken on
trust.

    python3 bench_agency.py --base-url URL --scenario <dir> --mode standard \
        --preset standard --model-name dsh-standard
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness, stats_from_events  # noqa: E402

# Graders that import the benchmark's own pins (openai, agent_sandbox,
# sii-agent-sdk) need the evaluation venv, not the system interpreter.
_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
GRADER_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

WORKSPACE_NOTICE = (
    "You are working in {workspace}, relative to your working directory {root}. "
    "Do not access outside paths. Implement the solution in C++17 strictly "
    "following the JSON specifications."
)

# Strings that only the grading side of this scenario uses. A tool call naming
# one of them reached the rubric, the scored artifacts, or the copy the grader
# makes, whichever directory it ran in. `Score_*` is deliberately absent: the
# task prompt itself names the rubric points, so an agent writing its own
# `Score_B` self-check is doing what it was asked, not reading the grader.
LEAK_MARKERS: tuple[tuple[str, str], ...] = (
    ("eval_task.py", "the grader's own source, which holds the rubric"),
    ("failed_points", "the rubric's failing points, as the grader reports them"),
    ("meta_eval", "the grader's scored output"),
    ("evalspace", "the directory the grader evaluates in"),
    ("description.json", "the scenario's task description"),
    ("AgencyBench", "the benchmark checkout itself"),
)

# First path segments that mean "the system's own tree", not an escape from the
# arm's workspace.
SYSTEM_DIRS = frozenset({
    "usr", "bin", "sbin", "lib", "libexec", "dev", "proc", "sys", "etc", "opt",
    "var", "tmp", "private", "System", "Library", "Applications", "Volumes",
})


# Another arm's private working root, by its generated name.
ARM_ROOT_PATTERN = re.compile(r"dsh-(?:work|grade)-[A-Za-z0-9_]{6,}")

# The durable session store, which holds every arm's own transcript.
SESSION_STORE_PATTERN = re.compile(r"/\.dsh/sessions/")


def defused(path: str) -> str:
    """Drop the `/private` prefix macOS reaches its temp dir through, so a path
    and the root it lives under compare equal whichever spelling they use."""
    return path[len("/private"):] if path.startswith("/private/") else path


def copy_tree(source: Path, target: Path) -> None:
    """Copy one workspace's contents into another directory."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def private_root(label: str) -> Path:
    """Create one private, randomly named working root under the system temp dir.

    Resolved, because macOS reaches the temp dir through the `/var` symlink:
    comparing an unresolved root against the paths the tools report would call
    every write inside the workspace an escape.
    """
    return Path(tempfile.mkdtemp(prefix=f"dsh-{label}-")).resolve()


def write_env_file(base: Path, model_name: str) -> Path:
    """Write the scenario's .env for one run (BYO agent, benchmark thresholds)."""
    env_file = base / f"env-{model_name}.env"
    env_file.write_text(
        f"SII_TARGET_MODEL={model_name}\n"
        "SII_AGENT_API_KEY=byo-harness-not-used\n"
        "SII_AGENT_API_BASE_URL=https://api.deepseek.com\n"
        "SII_USERNAME=byo\n"
        "SII_PASSWORD=byo\n"
        "SCORER_NAME=rubric\n"
        "MAX_SUBTASK_ATTEMPTS=3\n"
    )
    return env_file


def run_grader(
    scenario: Path,
    grade_root: Path,
    model_name: str,
    log: Path,
    grader: str = "eval-only",
) -> dict[str, object]:
    """Run the benchmark's own grader and return its parsed meta_eval.json.

    The grader derives its whole run tree from the description path's parent, so
    passing an absolute copy of that description outside the benchmark checkout
    keeps every scored artifact — and the evaluation workspaces the rubric runs
    commands in — out of the agent's reach.
    """
    env_file = write_env_file(grade_root, model_name)
    if grader == "shim":
        command = [GRADER_PYTHON, str(Path(__file__).resolve().parent / "eval_only_shim.py"),
                   str(scenario), env_file.name]
        cwd = Path(__file__).resolve().parent
    else:
        command = [GRADER_PYTHON, "eval_task.py", "--eval-only",
                   "--env", str(env_file), "--description", str(grade_root / "description.json")]
        cwd = scenario
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=5400)
    log.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr)
    meta_path = grade_root / model_name / "meta_eval.json"
    if not meta_path.exists():
        raise RuntimeError(f"grader produced no meta_eval.json; see {log}")
    return json.loads(meta_path.read_text())


def sync_to_grader(agent_root: Path, grade_root: Path, model_name: str, subtasks: int) -> None:
    """Place the arm's deliverables where the grader expects to read them."""
    for index in range(1, subtasks + 1):
        source = agent_root / f"subtask{index}" / "workspace" / "attempt_01"
        if not source.is_dir():
            continue
        target = grade_root / model_name / f"subtask{index}" / "workspace" / "attempt_01"
        target.mkdir(parents=True, exist_ok=True)
        copy_tree(source, target)


def audit_leakage(
    events: list[dict[str, object]],
    agent_root: Path,
    scenario: Path,
) -> dict[str, object]:
    """Measure whether the agent reached for the grading side of the scenario.

    The tool-call arguments the session log recorded are the evidence: a read,
    a grep, or a shell command that touched the rubric names its target there.
    Two things are counted — arguments mentioning a grading marker, and
    arguments naming a path outside the arm's own root.
    """
    markers: dict[str, int] = {}
    examples: list[str] = []
    outside = 0
    cross = 0
    calls = 0
    root = defused(str(agent_root))
    own_marker = Path(str(agent_root)).name
    for event in events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        arguments = data.get("arguments")
        if not isinstance(arguments, str):
            continue
        calls += 1
        hit = [name for name, _ in LEAK_MARKERS if name in arguments]
        for name in hit:
            markers[name] = markers.get(name, 0) + 1
        # An absolute path of two or more segments that is neither under the
        # arm's own root nor one of the system's own directories is the agent
        # naming a location outside its workspace (in practice: a scratch script
        # under /tmp). Single-segment tokens are skipped because this scenario's
        # own commands are slash-prefixed (`/history`).
        escaped = [
            token for token in re.findall(r"/(?:[A-Za-z0-9_.~+-]+/)+[A-Za-z0-9_.~+-]+", arguments)
            if not defused(token).startswith(root) and token.split("/")[1] not in SYSTEM_DIRS
        ]
        if escaped:
            outside += 1
        # Another arm's private root, or the durable session store: neither is
        # reachable under the arm's own seatbelt profile, so naming one is a
        # cross-arm reach whatever the outcome.
        other_roots = {name for name in ARM_ROOT_PATTERN.findall(arguments) if name != own_marker}
        if other_roots or SESSION_STORE_PATTERN.search(arguments):
            cross += 1
        if hit and len(examples) < 5:
            examples.append(f"{data.get('name')}: {arguments[:200]}")
    return {
        "tool_calls": calls,
        "markers": markers,
        "marker_calls": sum(markers.values()),
        "paths_outside_root_calls": outside,
        "cross_arm_calls": cross,
        "examples": examples,
        "scenario": str(scenario),
        "agent_root": str(agent_root),
    }


def recorder_usage(notes_root: Path, session_id: str) -> dict[str, object]:
    """Sum the note keeper's own model calls, which never enter the session log."""
    totals = {
        "calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    fields = {
        "input_tokens": "inputTokens",
        "output_tokens": "outputTokens",
        "cache_read_tokens": "cacheReadTokens",
        "cache_write_tokens": "cacheWriteTokens",
        "reasoning_tokens": "reasoningTokens",
    }
    path = notes_root / session_id / "recorder-usage.jsonl"
    if not path.exists():
        return totals
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        totals["calls"] += 1
        if not row.get("accepted"):
            totals["failed_calls"] += 1
        for total_key, row_key in fields.items():
            value = row.get(row_key)
            if isinstance(value, int):
                totals[total_key] += value
    return totals


def subtask_prompts(scenario: Path) -> list[str]:
    """Every subtask prompt of one scenario, in order."""
    description = json.loads((scenario / "description.json").read_text())
    count = int(description.get("subtask_count") or 0)
    return [description.get(f"subtask{index}", "") for index in range(1, count + 1)]


def feedback_for(meta: dict[str, object], subtask: str) -> str:
    """The rubric's failing points for one subtask, as the benchmark phrases them."""
    for item in meta.get("subtasks", []):  # type: ignore[union-attr]
        if not isinstance(item, dict) or item.get("name") != subtask:
            continue
        attempts = item.get("attempts") or []
        if not attempts:
            continue
        best = max(attempts, key=lambda attempt: attempt.get("score", 0))
        points = (best.get("rubric") or {}).get("failed_points") or []
        if points:
            bullets = "\n".join(f"- {point}" for point in points)
            return f"Focus on fixing these rubric gaps:\n{bullets}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--mode", required=True, help="label for this arm, e.g. standard")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--model-name", required=True, help="the grader's model id, and its run directory name")
    parser.add_argument("--max-subtasks", type=int, default=5)
    parser.add_argument("--second-pass", action="store_true", help="run the feedback attempt")
    parser.add_argument("--layout", choices=["subtasks", "flat"], default="subtasks",
                        help="subtasks: one workspace per subtask; flat: one workspace for the whole scenario")
    parser.add_argument("--grader", choices=["eval-only", "shim"], default="eval-only",
                        help="how to invoke the scenario's own grader")
    parser.add_argument("--seed-template", action="store_true",
                        help="stage the scenario's shipped workspace/ template before the first turn")
    parser.add_argument("--grade-only", action="store_true",
                        help="skip the agent turns and only re-run the grader on the existing workspace")
    parser.add_argument("--prompts-capture", action="store_true",
                        help="use the prompts the grader itself composes (captured through the shim) "
                             "instead of description.json — required for scenarios whose runner bakes "
                             "result-file paths and formats into its own prompt")
    parser.add_argument("--turn-timeout", type=float, default=2400.0)
    parser.add_argument("--turn-retries", type=int, default=3, help="resends of a turn that ended in an error")
    parser.add_argument("--agent-root", default="",
                        help="session cwd; created privately when omitted")
    parser.add_argument("--grade-root", default="",
                        help="the grader's base directory; created privately when omitted")
    parser.add_argument("--notes-root", default=str(Path.home() / ".dsh" / "notes"),
                        help="where the window keeps this session's notes and its usage log")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    scenario = Path(args.scenario).resolve()
    # Two unrelated private roots: the agent's own working tree, and the
    # grader's. Neither is inside the benchmark checkout, and neither is a
    # sibling of the other, so no relative path reaches the rubric.
    agent_root = Path(args.agent_root).resolve() if args.agent_root else private_root("work")
    grade_root = Path(args.grade_root).resolve() if args.grade_root else private_root("grade")
    agent_root.mkdir(parents=True, exist_ok=True)
    grade_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scenario / "description.json", grade_root / "description.json")
    # One artifact directory per arm: `meta_eval-pass*.json` and the grader logs
    # are named the same in every arm, so a shared directory would let the last
    # arm to finish overwrite the others' evidence.
    artifacts = Path(args.out).parent / args.model_name if args.out else grade_root / args.model_name
    artifacts.mkdir(parents=True, exist_ok=True)
    prompts = subtask_prompts(scenario)[: args.max_subtasks]
    if args.prompts_capture:
        # Ask the scenario's own coordinator to compose every subtask prompt.
        # Capturing by *running* the grader yields only the subtasks it reaches:
        # it skips every subtask after one fails, so subtask2 would never be sent.
        composed = artifacts / "composed-prompts.jsonl"
        completed = subprocess.run(
            [GRADER_PYTHON, str(Path(__file__).resolve().parent / "compose_prompts.py"),
             "--scenario", str(scenario), "--env", f"env-{args.model_name}.env",
             "--out", str(composed)],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=1800,
        )
        if completed.returncode != 0 or not composed.exists():
            raise RuntimeError(
                f"prompt composition failed for {scenario.name}: {completed.stderr[-400:]}"
            )
        rows = [json.loads(line) for line in composed.read_text().splitlines() if line.strip()]
        by_subtask = {row["subtask"]: row["prompt"] for row in rows}
        prompts = [by_subtask[f"subtask{index}"] for index in range(1, len(prompts) + 1)
                   if f"subtask{index}" in by_subtask]
        print(f"[INFO] composed {len(prompts)} grader-composed prompt(s)")

    harness = Harness(args.base_url)
    session_id = harness.create_session(str(agent_root), args.preset)
    report: dict[str, object] = {
        "mode": args.mode,
        "preset": args.preset,
        "scenario": str(scenario),
        "session_id": session_id,
        "run_root": str(agent_root),
        "agent_root": str(agent_root),
        "grade_root": str(grade_root),
        "turns": [],
    }
    print(f"[INFO] {args.mode}: session {session_id} on preset {args.preset} cwd {agent_root}")
    print(f"[INFO] {args.mode}: grader root {grade_root} (outside {scenario})")

    def turn_failed(events: list[dict[str, object]]) -> bool:
        """Whether the newest turn ended in an error rather than a real completion."""
        for event in reversed(events):
            if event.get("type") != "turn/end":
                continue
            reason = (event.get("data") or {}).get("reason") or {}
            return reason.get("kind") == "error"
        return False

    def send_turn(label: str, prompt: str, workspace: Path) -> dict[str, object]:
        """Send one user turn and record what it cost, retrying a failed turn.

        A transport or rate-limit failure ends the turn with `reason: error`;
        the work simply did not happen, so the same prompt is sent again instead
        of scoring a truncated attempt. Retries are recorded in the report.
        """
        started = time.time()
        attempts = 0
        while True:
            attempts += 1
            harness.prompt(session_id, prompt)
            idle = harness.wait_idle(session_id, timeout=args.turn_timeout)
            events = harness.read_log(session_id)
            if idle and not turn_failed(events):
                break
            if attempts > args.turn_retries:
                break
            print(f"[RETRY {label}] turn ended in error; resending (attempt {attempts + 1})")
            time.sleep(min(60.0, 10.0 * attempts))
        stats = stats_from_events(events, session_id)
        entry = {
            "label": label,
            "workspace": str(workspace),
            "idle": idle,
            "attempts": attempts,
            "seconds": round(time.time() - started, 1),
            "turns": stats.turns,
            "steps": stats.steps,
            "tool_calls": stats.tool_calls,
            "records": stats.records,
            "folds": stats.folds,
            "usage": stats.usage.as_dict(),
        }
        print(
            f"[TURN {label}] idle={idle} {entry['seconds']}s steps={stats.steps} "
            f"records={stats.records} folds={stats.folds} tokens={stats.usage.total_tokens}"
        )
        return entry

    def workspace_of(index: int) -> Path:
        """The workspace directory the agent produces, inside its own private root."""
        if args.layout == "flat":
            return agent_root / "workspace"
        return agent_root / f"subtask{index}" / "workspace" / "attempt_01"

    if args.seed_template:
        template = scenario / "workspace"
        if template.is_dir():
            copy_tree(template, workspace_of(1))
            print(f"[INFO] seeded workspace template from {template}")

    if args.grade_only:
        sync_to_grader(agent_root, grade_root, args.model_name, len(prompts))
        first = run_grader(scenario, grade_root, args.model_name, artifacts / "grader-pass1.log", args.grader)
        (artifacts / "meta_eval-pass1.json").write_text(json.dumps(first, ensure_ascii=False, indent=2))
        report["pass1"] = summarise(first)
        print(f"[SCORE grade-only] {json.dumps(report['pass1'], ensure_ascii=False)}")
        out = Path(args.out) if args.out else artifacts / f"run-{args.mode}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # Pass 1: one user turn per subtask, each seeded from the previous output.
    previous: Path | None = None
    for index, prompt in enumerate(prompts, start=1):
        workspace = workspace_of(index)
        workspace.mkdir(parents=True, exist_ok=True)
        if previous is not None and args.layout == "subtasks":
            copy_tree(previous, workspace)
        relative = workspace.relative_to(agent_root)
        full = f"{prompt}\n\n---\n{WORKSPACE_NOTICE.format(workspace=relative, root=agent_root)}"
        report["turns"].append(send_turn(f"subtask{index}", full, workspace))  # type: ignore[union-attr]
        previous = workspace

    # Score pass 1 with the benchmark's own grader, on a copy of the
    # deliverables: the grader's tree never touches the agent's.
    sync_to_grader(agent_root, grade_root, args.model_name, len(prompts))
    first = run_grader(scenario, grade_root, args.model_name, artifacts / "grader-pass1.log", args.grader)
    (artifacts / "meta_eval-pass1.json").write_text(json.dumps(first, ensure_ascii=False, indent=2))
    report["pass1"] = summarise(first)
    print(f"[SCORE pass1] {json.dumps(report['pass1'], ensure_ascii=False)}")

    if args.second_pass:
        for index, prompt in enumerate(prompts, start=1):
            feedback = feedback_for(first, f"subtask{index}")
            if not feedback:
                continue
            workspace = workspace_of(index)
            relative = workspace.relative_to(agent_root)
            full = (
                f"{prompt}\n\n{feedback}\n\n---\n"
                f"{WORKSPACE_NOTICE.format(workspace=relative, root=agent_root)}"
            )
            report["turns"].append(send_turn(f"subtask{index}-retry", full, workspace))  # type: ignore[union-attr]
        sync_to_grader(agent_root, grade_root, args.model_name, len(prompts))
        second = run_grader(scenario, grade_root, args.model_name, artifacts / "grader-pass2.log", args.grader)
        (artifacts / "meta_eval-pass2.json").write_text(json.dumps(second, ensure_ascii=False, indent=2))
        report["pass2"] = summarise(second)
        print(f"[SCORE pass2] {json.dumps(report['pass2'], ensure_ascii=False)}")

    events = harness.read_log(session_id)
    stats = stats_from_events(events, session_id)
    report["session"] = stats.as_dict()
    report["recorder"] = recorder_usage(Path(args.notes_root), session_id)
    report["leakage"] = audit_leakage(events, agent_root, scenario)
    out = Path(args.out) if args.out else artifacts / f"run-{args.mode}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[LEAK] {json.dumps(report['leakage'], ensure_ascii=False)[:400]}")
    print(f"[RECORDER] {json.dumps(report['recorder'], ensure_ascii=False)}")
    print(f"[OK] {args.mode} report -> {out}")
    return 0


def summarise(meta: dict[str, object]) -> dict[str, object]:
    """Reduce the grader's meta_eval.json to the numbers the comparison reports."""
    rows = []
    for item in meta.get("subtasks", []):  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        attempts = item.get("attempts") or []
        best = max(attempts, key=lambda attempt: attempt.get("score", 0)) if attempts else {}
        rows.append({
            "subtask": item.get("name"),
            "best_score": item.get("best_score", best.get("score")),
            "graded": bool(item.get("attempts")),
        })
    graded = [row for row in rows if row["graded"]]
    scores = [row["best_score"] for row in graded if isinstance(row["best_score"], (int, float))]
    total = sum(scores)
    return {
        "subtasks": rows,
        "attempted": len(graded),
        "total": total,
        "max": 10 * len(graded),
        "passes": len([score for score in scores if score >= 6]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
