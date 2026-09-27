"""Compose a scenario's own subtask prompts, without running anything.

Scenarios whose runner drives the agent directly also compose the prompt
themselves (`_compose_prompt`: workspace banner + task description + retry
feedback). A foreign agent should receive exactly that text, but capturing it by
running the grader only yields prompts for the subtasks the grader actually
reaches — it skips every subtask after one fails. This asks the coordinator
itself for each subtask's first-attempt prompt instead, with no model calls.

    python3 compose_prompts.py --scenario <dir> --env env-<model>.env --out prompts.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(scenario: Path):
    """Import a scenario's `eval_task.py` with the scenario as the working directory."""
    os.chdir(scenario)
    sys.path.insert(0, str(scenario))
    spec = importlib.util.spec_from_file_location("bench_eval_task", scenario / "eval_task.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {scenario / 'eval_task.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_eval_task"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scenario = Path(args.scenario).resolve()
    env_file = scenario / args.env
    module = load_module(scenario)

    env_data = module.load_env_file(env_file)
    env_config = module.EnvConfig.from_env(env_data)
    env_config.inject_defaults()
    coordinator = module.Task21Evaluator(env_config, scenario / "description.json") \
        if hasattr(module, "Task21Evaluator") else None
    if coordinator is None:
        # Bespoke scenarios name their coordinator differently; take the class
        # that owns `_compose_prompt`.
        for name, obj in vars(module).items():
            if isinstance(obj, type) and hasattr(obj, "_compose_prompt"):
                coordinator = obj(env_config, scenario / "description.json")
                print(f"[INFO] coordinator = {name}")
                break
    if coordinator is None:
        raise RuntimeError("no coordinator with _compose_prompt found")

    workspace = coordinator.workspace_dir  # type: ignore[attr-defined]
    description = coordinator.description  # type: ignore[attr-defined]
    count = int(description.get("subtask_count") or 0)
    written = 0
    out = Path(args.out)
    with out.open("w") as handle:
        for index in range(1, count + 1):
            content = description.get(f"subtask{index}")
            if not content:
                continue
            prompt = coordinator._compose_prompt(workspace, content, None)  # noqa: SLF001
            handle.write(json.dumps({"subtask": f"subtask{index}", "prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1
            print(f"[OK] subtask{index}: {len(prompt)} chars")
    print(f"[DONE] {written} prompt(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
