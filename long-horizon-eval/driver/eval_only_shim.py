"""Score an existing workspace with a benchmark grader that has no `--eval-only` flag.

Some AgencyBench scenarios ship only the original runner: it clears the
workspace, drives the SII agent, and then scores. To grade a workspace a
*foreign* agent produced, this shim imports the scenario's own `eval_task.py`
(unmodified), forces the SII agent to be unavailable, protects the existing
workspace from being cleared, and then calls the module's own `main()` — so the
rubric, thresholds and scoring code are exactly the benchmark's.

    python3 eval_only_shim.py <scenario_dir> <env_file_name>

The module is imported with the scenario directory as the current working
directory, because these graders resolve `judge_config.yaml` and similar files
relative to it.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import pathlib as _pathlib
import sys
from pathlib import Path

PRESERVE = ("workspace", "evalspace")


CAPTURE_PATH = os.environ.get("SHIM_CAPTURE_PROMPTS")


def _capture(prompt: object) -> None:
    """Record the prompt the coordinator would have sent, when capture is on."""
    if not CAPTURE_PATH or not isinstance(prompt, str):
        return
    import json as _json
    with open(CAPTURE_PATH, "a", encoding="utf-8") as handle:
        handle.write(_json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")


def _noop_send_sync(self, *args, **kwargs):
    """Stand-in for a synchronous agent turn the foreign agent already performed.

    With `SHIM_CAPTURE_PROMPTS` set, the prompt the coordinator WOULD have sent
    is appended to that file, so a foreign agent can be given the benchmark's own
    prompt (result-file paths, loss thresholds) instead of a paraphrase.
    """
    if args:
        _capture(args[0])
    return ""


async def _noop_send_async(self, *args, **kwargs):
    """Stand-in for an awaited agent turn the foreign agent already performed."""
    if args:
        _capture(args[0])
    return None


def _completed_none():
    """A coroutine that yields None, for coordinators awaiting `send`."""
    return _noop_send(None)


def is_protected(path: os.PathLike[str] | str, scenario: Path) -> bool:
    """Whether a deletion target is a workspace this shim must keep."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    if scenario not in resolved.parents:
        return False
    return resolved.name in PRESERVE or resolved.parent.name in PRESERVE


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: eval_only_shim.py <scenario_dir> <env_file_name>")
        return 2
    scenario = Path(sys.argv[1]).resolve()
    env_file = sys.argv[2]

    os.chdir(scenario)
    sys.path.insert(0, str(scenario))
    spec = importlib.util.spec_from_file_location("bench_eval_task", scenario / "eval_task.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {scenario / 'eval_task.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_eval_task"] = module
    spec.loader.exec_module(module)

    # 1. The SII agent must never run: the workspace already holds the agent's work.
    patched_agents = 0
    for name, obj in vars(module).items():
        if isinstance(obj, type) and hasattr(obj, "_sdk_available"):
            obj._sdk_available = lambda self: False  # type: ignore[method-assign]
            patched_agents += 1

    # 1b. Coordinators that drive an agent class directly: their `send` result is
    #     either ignored or used as display text, so a no-op coroutine is enough.
    patched_send = 0
    for name, obj in vars(module).items():
        if not isinstance(obj, type):
            continue
        attributes = vars(obj)
        if "send" in attributes and ("Runner" in name or "Agent" in name):
            # Awaiting is the coordinator's choice, so the stand-in must match it.
            obj.send = _noop_send_async if inspect.iscoroutinefunction(attributes["send"]) else _noop_send_sync
            patched_send += 1

    # 2. Some coordinators restage a template over the workspace before running
    #    (`_reset_model_workspace`, `_reset_workspace`, …). Neutralize those too.
    reset_patched = 0
    for name, obj in vars(module).items():
        if not isinstance(obj, type):
            continue
        for attribute in list(vars(obj)):
            if attribute.startswith("_reset") and "workspace" in attribute:
                setattr(obj, attribute, lambda self, *a, **k: None)
                reset_patched += 1
            elif attribute in {"_stage_workspace", "_prepare_workspace", "_copy_workspace"}:
                setattr(obj, attribute, lambda self, *a, **k: None)
                reset_patched += 1

    # 3. The existing workspace must survive: neutralize clearing and deletion.
    def preserved(name: str, original):
        def wrapper(path, *args, **kwargs):
            if is_protected(path, scenario):
                print(f"[SHIM] preserving {path} (skipping {name})")
                return None
            return original(path, *args, **kwargs)
        return wrapper

    original_unlink = _pathlib.Path.unlink

    def guarded_unlink(self, *args, **kwargs):
        if is_protected(self, scenario):
            print(f"[SHIM] preserving {self} (skipping unlink)")
            return None
        return original_unlink(self, *args, **kwargs)

    _pathlib.Path.unlink = guarded_unlink  # type: ignore[method-assign]

    # Some scenarios stage their shipped `source/` tree over the workspace at the
    # start of every attempt — including in eval-only mode, where that silently
    # overwrites the foreign agent's edits with the pristine template. Copies
    # INTO a workspace are refused; the workspace -> evalspace copy the grader
    # scores with is untouched.
    if hasattr(module, "copy_workspace"):
        original_copy = module.copy_workspace

        def copy_guarded(source, destination, *args, **kwargs):
            if is_protected(destination, scenario) and not is_protected(source, scenario):
                print(f"[SHIM] refusing to restage {source} over {destination}")
                return None
            return original_copy(source, destination, *args, **kwargs)

        module.copy_workspace = copy_guarded

    if hasattr(module, "clear_directory"):
        module.clear_directory = preserved("clear_directory", module.clear_directory)  # type: ignore[assignment]
    import shutil as _shutil
    if getattr(module, "shutil", None) is not None:
        module.shutil.rmtree = preserved("rmtree", _shutil.rmtree)  # type: ignore[attr-defined]

    print(f"[SHIM] scenario={scenario.name} sdk probes={patched_agents} send={patched_send} reset={reset_patched}")
    sys.argv = ["eval_task.py", "--env", env_file]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
