#!/usr/bin/env python3
"""Drive one UltraHorizon environment with a DeepSeek Harness session as the agent.

The benchmark's own entry point (``bench/UltraHorizon/interface.py``) hardcodes
OpenManus: it scans the environment for ``[agent tool]`` methods, exposes them as
OpenAI function-calling tools, and runs ``Manus`` until the environment commits.
This adapter keeps the environment untouched and replaces only the agent side:

* one DSH session per episode, kept alive for the whole episode, so every
  environment observation is a separate user turn and the episode is a genuine
  multi-turn long-horizon conversation;
* the environment's own async tool methods are called directly (``move``,
  ``input_sequences``, ``conduct_cross``, ...), so scores, validation, step caps
  and the commit gate are the benchmark's, not a reimplementation;
* determinism comes from seeding Python's global ``random`` (all three envs draw
  from it) via the reversible in-process patch described in ``seed_scope``;
* the LLM judge uses the prompt template that is defined inside each
  environment's ``commit_final_result`` and is re-pointed at an
  OpenAI-compatible endpoint (DeepSeek by default) by a reversible patch of the
  ``OpenAI`` symbol the env modules imported.

Nothing under ``bench/UltraHorizon`` is read at import time except the source of
the judge template; nothing there is ever written.

Example:
    python3 ultrahorizon_driver.py --env seq --steps 3 --seed 7 \
        --preset standard --base-url http://127.0.0.1:58422 --out /tmp/ep.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent                       # long-horizon-eval/
BENCH_ROOT = REPO_ROOT / "bench" / "UltraHorizon"
RESULTS_ROOT = REPO_ROOT / "runs" / "ultrahorizon"

sys.path.insert(0, str(HERE))
from dsh import Harness, SessionStats, stats_from_events  # noqa: E402

# --------------------------------------------------------------------------
# Environment imports (no OpenManus, no interface.py)
# --------------------------------------------------------------------------


def _import_envs() -> dict[str, Any]:
    """Import the three env classes directly from the cloned benchmark repo.

    ``envs/*/env.py`` do ``from envs.common import Difficulty``, so the bench root
    must be importable as a package root. That root also contains ``interface.py``,
    which imports OpenManus; importing ``envs.*`` never touches it.
    """
    if not BENCH_ROOT.is_dir():
        raise SystemExit(f"benchmark repo not found: {BENCH_ROOT}")
    # Importing the envs would otherwise drop __pycache__/*.cpython-<ver>.pyc files
    # into the cloned repo, which we must leave untouched.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BENCH_ROOT))
    from envs.bio_env.env import GeneticsLabEnvironment
    from envs.common import Difficulty
    from envs.grid_env.env import MysteryGridEnvironment
    from envs.seq_env.env import SequenceExploreEnvironment

    return {
        "seq": SequenceExploreEnvironment,
        "grid": MysteryGridEnvironment,
        "bio": GeneticsLabEnvironment,
        "Difficulty": Difficulty,
    }


# --------------------------------------------------------------------------
# Determinism: reversible patching of the global RNG the envs use
# --------------------------------------------------------------------------


def seed_scope(seed: int) -> Callable[[], None]:
    """Pin Python's global ``random`` to ``seed`` and return an undo function.

    All three environments draw from the *global* ``random`` module:

    * ``seq``  - deterministic already (no RNG use at all);
    * ``grid`` - ``random.choice`` when generating the grid and ``random.randint``
      in the ``GameState`` default factory; ``reset`` regenerates both;
    * ``bio``  - calls ``random.seed(seed)`` in ``__init__`` and then uses
      ``random.choice``/``random.shuffle`` for gamete segregation.

    ``bio`` is therefore reproducible on its own, but only for whatever seed the
    caller passes, while ``grid`` inherits global state and is not reproducible.
    We patch ``random.seed`` so that any reseed performed *by the environment*
    (i.e. ``random.seed(N)`` from env code or ``random.seed(None)`` from anywhere)
    is forced back to the episode seed. This keeps ``bio``'s documented
    ``seed=`` contract intact while making ``grid`` reproducible as well.

    The patch is deliberately small, clearly scoped to the episode, and undone by
    the returned callable. It does not modify the cloned repository.
    """
    original_seed = random.seed
    random.seed(seed)

    def pinned(a: Any = None, version: int = 2) -> None:
        original_seed(seed if a is None else a)

    random.seed = pinned  # type: ignore[assignment]

    def undo() -> None:
        random.seed = original_seed  # type: ignore[assignment]

    return undo


# --------------------------------------------------------------------------
# Judge: reuse the env's own template, re-point its client at our endpoint
# --------------------------------------------------------------------------

JUDGE_PROMPT_MARKER = 'prompt = """'
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_JUDGE_MODEL = "deepseek-chat"
CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"


def load_api_key() -> str:
    """Read ``DEEPSEEK_API_KEY`` from the environment, else the DSH credentials file."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    if CREDENTIALS.is_file():
        for line in CREDENTIALS.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise SystemExit(
        "no DEEPSEEK_API_KEY in the environment and none found in "
        f"{CREDENTIALS}; the judge cannot be reached"
    )


def env_judge_template(env_cls: type) -> tuple[str, str]:
    """Extract the literal judge prompt and the ground-truth payload from env source.

    Every environment defines its judging rubric as an inline ``prompt = \"\"\"...\"\"\"``
    literal inside ``commit_final_result``, followed by
    ``prompt.replace("{ground_truth}", ...)`` or ``prompt.replace("{content}", ...)``.
    Rather than inventing a rubric we lift the repo's own text verbatim, and we take
    the ground truth from the environment instance's own difficulty mapping.
    """
    src = Path(sys.modules[env_cls.__module__].__file__ or "").read_text(encoding="utf-8")
    start = src.find(JUDGE_PROMPT_MARKER)
    if start < 0:
        raise SystemExit(f"{env_cls.__name__}: no judge prompt literal found in its source")
    # Tokenize the remainder and take the first string literal: that is the template
    # itself. Robust to indentation and to closing quotes indented differently from
    # the body (which is how all three envs are written).
    import tokenize

    reader = io.StringIO(src[start + len("prompt = "):]).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.STRING:
            prompt = ast.literal_eval(token.string)  # type: ignore[arg-type]
            break
    else:
        raise SystemExit(f"{env_cls.__name__}: judge prompt literal is not a plain string")
    return prompt, src


def env_ground_truth(env: Any, env_name: str) -> dict[str, Any]:
    """The env's own ground-truth mapping for the current difficulty."""
    if env_name == "seq":
        # Mirrors the rule_descriptions table in envs/seq_env/env.py.
        if env.difficulty.name == "EASY":
            return {
                "rule_1": "Alternate characters from main and vice strings (simplified interleaving)",
                "rule_2": "Add character position values (main[i] + vice[i]) with proper modular arithmetic",
                "rule_3": "Take maximum character at each position between main and vice, or remove consecutive duplicates if no main/vice",
                "rule_4": "Take main if main[i] >= vice[i], else take vice[i], or append length character if no main/vice",
                "rule_5": "Combine main and vice strings then sort alphabetically, or sort current string if no main/vice",
            }
        return {
            "rule_1": "Interweave main and vice characters completely (5 points). The leading sequence is determined by the parity of current step number: main leads if odd, vice leads if even. Equivalent wording allowed: 'alternating main and vice characters', 'merge main and vice with step-based leading character'. (15 points)",
            "rule_2": "Reverse the current sequence and shift all characters forward by n (n = current step number) positions in the alphabet (cyclic A→B→...→Z→A). Concatenate this with the similarly shifted version of the original sequence. Equivalent wording allowed: 'flip sequence', 'letter shift', 'add letter value with current step number and then modulo 26' (20 points)",
            "rule_3": "Select the character at position (current step number mod 10, with 0 treated as 10). Copy this character 'a' times (where a is that current step number modulo 10). Append the copies to the current sequence. (20 points)",
            "rule_4": "Take the first 5 characters of the current sequence and combine them character-wise with the last main sequence by adding their alphabet positions modulo 26. Append unchanged remainder of the current sequence after position 5. Equivalent wording allowed: 'sum first 5 characters with last main sequence modulo alphabet'. (20 points)",
            "rule_5": "If current step number is prime, find the most frequent letter in the current sequence and replace all its occurrences with the next letter in the alphabet (cyclic A→B→...→Z→A). Otherwise leave the sequence unchanged. Equivalent wording allowed: 'replace most common character with its next letter on prime steps', 'frequency-based substitution of most frequent character when step is prime'. (20 points)",
        }
    if env_name == "grid":
        descriptions = {
            "effect_1": "Simple +1 score",
            "effect_2": "Simple +1 energy",
            "effect_3": "Position-based: odd coordinates (x+y) +1 score, even coordinates (x+y) -1 score",
            "effect_4": "Simple -1 score",
            "effect_5": "Energy threshold effect: if energy >= 15 then +2 score, if energy < 15 then -2 score",
            "effect_6": "Step count effect: if steps % 3 == 0 then +2 score, else -1 score",
            "effect_7": "Corner and edge effects: corners give +3 score, edges give +3 score, otherwise no effect. Formally, if x in (0, 1, 8, 9) or y in (0, 1, 8, 9): score+=3",
            "effect_8": "Visit count based effect: +1 score if this letter type has been visited odd number of times, 0 score if even number of times",
            "effect_9": "Energy management: if energy is low then -2 score and +10 energy, if energy is high then +1 score",
            "effect_10": "Complex position calculation: score change = x - y",
        }
        mapping = env.DIFFICULTY_EFFECTS[env.difficulty]
        return {letter: descriptions[name] for letter, name in mapping.items()}
    if env_name == "bio":
        # envs/bio_env/env.py hardcodes its rubric text; the ground truth is get_ground_truth().
        return env.get_ground_truth()
    raise SystemExit(f"no ground truth extractor for env {env_name!r}")


def reroute_judge_client(env_module: Any, api_key: str, base_url: str, model: str) -> Callable[[], None]:
    """Re-point the env module's ``OpenAI`` symbol at our endpoint. Reversible.

    ``envs/*/env.py`` do ``from openai import OpenAI`` and build the judge client
    from the placeholder ``judge_config.yaml`` shipped in the repo. We swap that
    module-global with a factory that ignores the placeholder credentials and uses
    the real endpoint, so the env's own ``commit_final_result`` performs the
    judging with its own rubric and its own ground truth.
    """
    import openai as openai_module

    original = env_module.OpenAI

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        return openai_module.OpenAI(api_key=api_key, base_url=base_url)

    env_module.OpenAI = factory

    def undo() -> None:
        env_module.OpenAI = original

    return undo


def parse_judge_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a judge reply (mirrors the env's own cleaning)."""
    cleaned = text.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": cleaned}


def patch_placeholder_judge_config(env_cls: type, api_key: str, base_url: str, model: str) -> Callable[[], None]:
    """Make the env's ``__init__`` stop needing ``./judge_config.yaml``. Reversible.

    ``load_judge_config`` opens the literal relative path ``judge_config.yaml`` from
    the *process* cwd, so constructing any environment fails unless that file sits
    in the working directory (and the shipped one holds placeholder credentials).
    We replace the classmethod on the imported class only - no file in the cloned
    repo is touched - and give it the real endpoint. When judging is disabled we
    hand back the placeholder values, which keeps ``__init__`` working while
    ``commit_final_result`` still ends in the env's own "Evaluation failed" path.
    """
    original = env_cls.load_judge_config
    env_cls.load_judge_config = classmethod(  # type: ignore[assignment]
        lambda cls: {"model": model, "base_url": base_url, "api_key": api_key}
    )

    def undo() -> None:
        env_cls.load_judge_config = original  # type: ignore[assignment]

    return undo


def run_judge(
    env: Any,
    env_name: str,
    env_cls: type,
    answer: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Score one finished episode with the repo's own rubric over an OpenAI-compatible API."""
    template, _src = env_judge_template(env_cls)
    prompt = template.replace("{content}", answer)
    if "{ground_truth}" in prompt:
        prompt = prompt.replace(
            "{ground_truth}", json.dumps(env_ground_truth(env, env_name), indent=2)
        )
    import openai as openai_module

    client = openai_module.OpenAI(api_key=api_key, base_url=base_url)
    started = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise evaluator of rules."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = getattr(response.choices[0].message, "content", "") or ""
    usage = getattr(response, "usage", None)
    return {
        "model": model,
        "base_url": base_url,
        "template_source": str(env_cls.__module__),
        "prompt_chars": len(prompt),
        "seconds": round(time.time() - started, 1),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "raw_output": text,
        "result": parse_judge_json(text),
    }


# --------------------------------------------------------------------------
# Action protocol
# --------------------------------------------------------------------------


@dataclass
class Action:
    """One parsed agent decision."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


ACTION_INSTRUCTION = """You are the agent inside a benchmark environment. Each user message is one \
environment observation. Reply with EXACTLY ONE JSON object and nothing else - no markdown fence, \
no prose before or after it. Do not use any tools; do not read or write files; the environment is \
only reachable through the JSON action.

JSON action forms:
  {"action": "move", "direction": "up"}                  # one tile per action
  {"action": "input_sequences", "main_sequence": "ABCDE", "vice_sequence": "EDCBA"}
  {"action": "conduct_cross", "parent1_id": 1, "parent2_id": 2, "num_offspring": 10}
  {"action": "query_organisms", "start_id": 1, "end_id": 20}
  {"action": "get_current_state"}
  {"action": "get_full_map"}
  {"action": "get_lab_status"}
  {"action": "get_game_info"}
  {"action": "get_history", "last_n": 5}
  {"action": "reset"}
  {"action": "commit", "content": "<your full final answer, newlines encoded as \\n>"}

The episode ends after the "commit" action or when the step budget is exhausted, so commit only \
once you are confident. A "commit" before the environment's required step count is reached is \
rejected by the environment and does not end the episode or produce a score."""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the last JSON object in a model reply, fence-tolerant."""
    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        candidates.append(match.group(1))
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start:index + 1])
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def quiet_stdout(enabled: bool = True) -> Any:
    """Swallow the environments' own ``print`` chatter (they dump the whole prompt).

    ``envs/*/env.py`` print the full environment prompt, the judge config and the
    evaluation prompt. That is noise for a driver whose real output is the result
    JSON, so by default it is redirected into an in-memory buffer. Nothing is
    suppressed under ``--verbose``.
    """
    if not enabled:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


# Field names a model plausibly uses for its final answer when the env's own
# tool schema calls the argument `content`. Accepting them keeps the comparison
# measuring the context policy instead of the model's choice of JSON key.
COMMIT_FIELDS = ("content", "rules", "answer", "report", "result", "description", "text", "summary")


def commit_content(args: dict[str, Any]) -> str:
    """The text one commit action submits, from `content` or any equivalent field.

    A stringified Python list (a common model habit for `rules`) is unrolled into
    one line per element; anything else is passed through verbatim.
    """
    for field in COMMIT_FIELDS:
        value = args.get(field)
        if isinstance(value, str) and value.strip():
            return _unroll(value)
    joined = [value for value in args.values() if isinstance(value, str) and value.strip()]
    return _unroll("\n".join(joined))


def _unroll(value: str) -> str:
    """Turn a stringified list into newline-separated text when it parses as one."""
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return value
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, (list, tuple)) and all(isinstance(item, str) for item in parsed):
        return "\n".join(parsed)
    return value


def parse_action(text: str) -> Action | None:
    """Turn one agent reply into an Action, or None when no JSON object is present."""
    payload = extract_json_object(text)
    if payload is None:
        return None
    name = payload.get("action") or payload.get("tool")
    if not isinstance(name, str):
        return None
    if name in {"commit", "commit_final_result"}:
        payload.setdefault("content", payload.get("answer") or payload.get("report") or "")
    args = {key: value for key, value in payload.items() if key not in {"action", "tool"}}
    return Action(name=name, args=args, raw=text)


# --------------------------------------------------------------------------
# Per-environment tool surface
# --------------------------------------------------------------------------


class EnvBridge:
    """Uniform call surface over the three environments' async tool methods."""

    def __init__(self, env: Any, env_name: str, verbose: bool = False) -> None:
        self.env = env
        self.env_name = env_name
        self.verbose = verbose

    def _run(self, coroutine: Any) -> Any:
        with quiet_stdout(not self.verbose):
            return asyncio.run(coroutine)

    def observe(self) -> dict[str, Any]:
        """The environment's opening observation for step 1."""
        if self.env_name == "grid":
            return {"observation": "episode_start", "game_state": self._run(self.env.get_current_state())}
        if self.env_name == "seq":
            return {"observation": "episode_start", "game_info": self._run(self.env.get_game_info())}
        return {"observation": "episode_start", "lab_status": self._run(self.env.get_lab_status())}

    def action_names(self) -> list[str]:
        if self.env_name == "seq":
            return ["input_sequences", "get_game_info", "get_history", "commit"]
        if self.env_name == "grid":
            return ["move", "get_current_state", "get_full_map", "reset", "commit"]
        return ["conduct_cross", "query_organisms", "remove_organisms", "get_lab_status", "commit"]

    def call(self, action: Action) -> dict[str, Any]:
        """Execute one parsed action against the environment."""
        name = action.name
        args = action.args
        env = self.env
        if name in {"commit", "finish", "submit"}:
            return self._run(env.commit_final_result(commit_content(args)))
        if name == "move":
            direction = args.get("direction") or args.get("dir")
            if isinstance(direction, list):
                results = [self._run(env.move(d)) for d in direction]
                return {"success": True, "moves": results}
            return self._run(env.move(str(direction)))
        if name in {"get_current_state", "get_full_map", "get_game_info", "get_lab_status", "reset"}:
            return self._run(getattr(env, name)())
        if name == "get_history":
            last_n = args.get("last_n")
            return self._run(env.get_history(last_n if isinstance(last_n, int) else None))
        if name == "input_sequences":
            return self._run(
                env.input_sequences(str(args.get("main_sequence", "")), str(args.get("vice_sequence", "")))
            )
        if name == "conduct_cross":
            return self._run(
                env.conduct_cross(
                    int(args.get("parent1_id", 0)),
                    int(args.get("parent2_id", 0)),
                    int(args.get("num_offspring", 10)),
                )
            )
        if name == "query_organisms":
            end_id = args.get("end_id")
            return self._run(
                env.query_organisms(int(args.get("start_id", 1)), int(end_id) if end_id is not None else None)
            )
        if name == "remove_organisms":
            ids = args.get("organism_ids") or args.get("ids") or []
            return self._run(env.remove_organisms([int(i) for i in ids]))
        return {
            "success": False,
            "error": f"unknown action {name!r}; available: {', '.join(self.action_names())}",
        }

    def is_done(self) -> bool:
        return bool(getattr(self.env, "committed", False))

    def progress(self) -> dict[str, Any]:
        """The environment's own step counter, whatever it is called."""
        for attribute in ("total_steps", "current_experiments"):
            value = getattr(self.env, attribute, None)
            if isinstance(value, int):
                return {"counter": attribute, "value": value}
        return {}


# --------------------------------------------------------------------------
# Episode loop
# --------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    """Run one full episode: one env, one DSH session, one result dict."""
    envs = _import_envs()
    env_cls = envs[args.env]
    difficulty = envs["Difficulty"][args.difficulty.upper()]
    seed = args.seed if args.seed is not None else 0

    cwd = Path(args.cwd).resolve() if args.cwd else (RESULTS_ROOT / f"{args.env}-seed{seed}-{args.preset}")
    cwd.mkdir(parents=True, exist_ok=True)

    harness = Harness(args.base_url)
    result: dict[str, Any] = {
        "env": args.env,
        "seed": seed,
        "preset": args.preset,
        "base_url": args.base_url,
        "difficulty": args.difficulty,
        "step_cap": args.steps,
        "agent_turn_cap": args.agent_turns,
        "cwd": str(cwd),
        "started_at": _now(),
        "status": "error",
        "finished": False,
        "steps": 0,
        "session_id": None,
        "commands": [],
        "notes": [],
    }

    original_seed = random.seed
    env_module = sys.modules[env_cls.__module__]
    undo_seed: Callable[[], None] = lambda: None
    undo_judge: Callable[[], None] = lambda: None
    undo_config: Callable[[], None] = lambda: None

    try:
        # --- determinism, then the environment ---------------------------
        undo_seed = seed_scope(seed)
        result["notes"].append(
            "random.seed pinned to the episode seed for the whole episode "
            "(reversible patch of the global RNG; no repo file was modified)"
        )
        api_key = load_api_key() if args.judge else "placeholder"
        undo_config = patch_placeholder_judge_config(
            env_cls, api_key, args.judge_base_url, args.judge_model
        )
        result["notes"].append(
            "load_judge_config replaced on the imported class so the env stops reading "
            "the relative path ./judge_config.yaml (reversible; the cloned repo is read-only)"
        )

        with quiet_stdout(not args.verbose):
            env = env_cls(difficulty=difficulty, required_steps=args.steps)
        if args.judge:
            undo_judge = reroute_judge_client(env_module, api_key, args.judge_base_url, args.judge_model)
            result["notes"].append(
                f"env module OpenAI symbol re-pointed to {args.judge_base_url} / {args.judge_model} "
                "so commit_final_result judges with the repo's own rubric (reversible)"
            )
        else:
            result["notes"].append(
                "judging disabled (--judge not given): commit_final_result will report the "
                "placeholder judge config as an evaluation failure"
            )

        bridge = EnvBridge(env, args.env, verbose=args.verbose)

        # --- session -----------------------------------------------------
        session_id = harness.create_session(cwd=str(cwd), preset=args.preset)
        result["session_id"] = session_id

        transcript: list[dict[str, Any]] = []
        used_actions: dict[str, int] = {}
        first_message = (
            f"{env.env_prompt}\n\n---\n{ACTION_INSTRUCTION}\n\n"
            f"Available actions for this environment: {', '.join(bridge.action_names())}.\n"
            f"The environment accepts a commit once {args.steps} accepted steps are reached, and you "
            f"have at most {args.agent_turns} turns in total.\n\n"
            f"First observation:\n{json.dumps(bridge.observe(), indent=2)}"
        )
        pending = first_message
        started_wall = _now()

        for step in range(1, args.agent_turns + 1):
            harness.prompt(session_id, pending)
            if not harness.wait_idle(session_id, timeout=args.turn_timeout):
                result["notes"].append(f"turn {step} timed out after {args.turn_timeout}s")
                break
            stats = stats_from_events(harness.read_log(session_id), session_id)
            reply = stats.last_assistant_text
            action = parse_action(reply)
            entry: dict[str, Any] = {
                "step": step,
                "required_steps": args.steps,
                "agent_turns": args.agent_turns,
                "env_steps_used": bridge.progress().get("value"),
                "reply_chars": len(reply),
                "reply_head": reply[:400],
                "action": action.name if action else None,
                "action_args": (
                    {k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in action.args.items()}
                    if action
                    else None
                ),
            }
            transcript.append(entry)

            if action is None:
                entry["observation"] = {"error": "no JSON action found in the reply"}
                pending = (
                    "Your previous reply contained no JSON action. Reply with exactly one JSON "
                    f"object, for example {{\"action\": \"{bridge.action_names()[0]}\"}}. "
                    f"Available actions: {', '.join(bridge.action_names())}."
                )
                result["steps"] = step
                if step >= args.agent_turns:
                    result["notes"].append("no JSON action in the final step of the budget")
                continue

            used_actions[action.name] = used_actions.get(action.name, 0) + 1
            try:
                observation = bridge.call(action)
            except Exception as error:  # env tool raised; report it as an observation
                observation = {"success": False, "error": f"{type(error).__name__}: {error}"}
            entry["observation"] = observation
            result["steps"] = step

            if bridge.is_done():
                result["finished"] = True
                result["notes"].append(f"environment committed at step {step}")
                break
            pending = json.dumps(observation, indent=2, default=str)

        result["wall_seconds"] = round(_now() - started_wall, 1)
        result["action_counts"] = used_actions
        result["transcript"] = transcript
        result["env_counter"] = bridge.progress()
        result["committed"] = bool(getattr(env, "committed", False))
        # The grid env's in-run score is a reward signal, not the benchmark metric
        # (the metric is the judge's final_score). Record it when the env exposes it.
        for entry in reversed(transcript):
            observation = entry.get("observation")
            if isinstance(observation, dict) and isinstance(observation.get("score"), int):
                result["env_reward"] = {
                    "score": observation["score"],
                    "energy": observation.get("energy"),
                    "at_step": entry["step"],
                }
                break

        # env-computed score, if the judge ran
        final_result = getattr(env, "final_result", {}) or {}
        result["final_result"] = final_result
        judge_result = final_result.get("judge_result") if isinstance(final_result, dict) else None
        if isinstance(judge_result, dict):
            result["final_score"] = judge_result.get("final_score")
            result["score_breakdown"] = judge_result.get("score_breakdown")
        result["agent_final_answer"] = (
            final_result.get("judge_input") if isinstance(final_result, dict) else None
        ) or (
            transcript[-1]["reply_head"] if transcript else None
        )

        # --- token accounting -------------------------------------------
        events = harness.read_log(session_id)
        stats: SessionStats = stats_from_events(events, session_id)
        result["stats"] = stats.as_dict()
        result["status"] = "ok"

        if args.judge and not result.get("committed"):
            result["host_judge"] = run_judge(
                env,
                args.env,
                env_cls,
                str(result.get("agent_final_answer") or ""),
                load_api_key(),
                args.judge_base_url,
                args.judge_model,
            )
            host_judge_result = result["host_judge"]["result"]
            if isinstance(host_judge_result, dict):
                result["final_score"] = host_judge_result.get("final_score")
                result["score_breakdown"] = host_judge_result.get("score_breakdown")
    except Exception as error:  # keep a partial result file even on failure
        result["error"] = f"{type(error).__name__}: {error}"
        result["status"] = "error"
    finally:
        undo_judge()
        undo_config()
        undo_seed()
        result["random_seed_restored"] = random.seed is original_seed
        result["ended_at"] = _now()
    return result


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """Write one episode result as JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """CLI surface for one episode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["seq", "grid", "bio"], required=True)
    parser.add_argument("--steps", type=int, default=8, help="required environment steps (the env's commit gate)")
    parser.add_argument(
        "--agent-turns", type=int, default=0,
        help="agent turns allowed in the episode; 0 means 4x --steps, matching the repo's own "
             "max_run_steps for grid/seq (8x for bio). The loop cap must exceed required_steps, "
             "otherwise the agent runs out of turns before it can commit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="standard")
    parser.add_argument("--base-url", default="http://127.0.0.1:58422")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--cwd", default=None, help="agent workspace; defaults to runs/ultrahorizon/<env>-seed<seed>-<preset>")
    parser.add_argument("--turn-timeout", type=float, default=1800.0, help="seconds to wait for one agent turn")
    parser.add_argument("--judge", action="store_true", help="score the episode with the repo's judge rubric")
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--verbose", action="store_true", help="let the environments' own prints through")
    parser.add_argument(
        "--judge-only",
        type=Path,
        default=None,
        metavar="RESULT_JSON",
        help="skip the episode: re-score a finished result JSON with the repo's rubric",
    )
    return parser


def judge_existing(path: Path, args: argparse.Namespace) -> int:
    """Score one already-written episode result with the repo's judge rubric.

    This is the cheap path for scoring a finished episode without replaying it:
    it needs the answer and the env/difficulty the episode ran under, both of
    which are in the result JSON, plus the judge endpoint.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    envs = _import_envs()
    env_name = payload.get("env")
    env_cls = envs[env_name]
    patch_placeholder_judge_config(env_cls, "placeholder", "http://127.0.0.1:1/v1", "none")
    undo_seed = seed_scope(int(payload.get("seed") or 0))
    try:
        with quiet_stdout(not args.verbose):
            env = env_cls(
                difficulty=envs["Difficulty"][str(payload.get("difficulty", "hard")).upper()],
                required_steps=int(payload.get("step_cap") or 1),
            )
        verdict = run_judge(
            env,
            env_name,
            env_cls,
            str(payload.get("agent_final_answer") or ""),
            load_api_key(),
            args.judge_base_url,
            args.judge_model,
        )
    finally:
        undo_seed()
    result = verdict["result"]
    payload["final_score"] = result.get("final_score") if isinstance(result, dict) else None
    payload["score_breakdown"] = result.get("score_breakdown") if isinstance(result, dict) else None
    payload["judge"] = {
        "model": verdict["model"],
        "base_url": verdict["base_url"],
        "template_source": verdict["template_source"],
        "prompt_chars": verdict["prompt_chars"],
        "seconds": verdict["seconds"],
        "usage": verdict["usage"],
        "result": result,
    }
    write_result(path, payload)
    print(json.dumps({
        "env": env_name,
        "final_score": payload["final_score"],
        "judge_usage": verdict["usage"],
        "judge_seconds": verdict["seconds"],
        "out": str(path),
    }, indent=2, default=str), flush=True)
    return 0


def main() -> int:
    """Run one episode (or judge one finished episode) and write its result JSON."""
    args = build_parser().parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.agent_turns <= 0:
        args.agent_turns = args.steps * (8 if args.env == "bio" else 4)
    if args.judge_only:
        return judge_existing(args.judge_only, args)
    try:
        result = run_episode(args)
    except Exception as error:
        payload = {
            "env": args.env,
            "seed": args.seed,
            "preset": args.preset,
            "step_cap": args.steps,
            "agent_turn_cap": args.agent_turns,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
        write_result(args.out, payload)
        print(json.dumps(payload, indent=2), flush=True)
        return 1
    write_result(args.out, result)
    usage = (result.get("stats") or {}).get("usage", {})
    summary = {
        key: result.get(key)
        for key in ("env", "seed", "preset", "status", "finished", "steps", "wall_seconds", "session_id")
    }
    # Spelled out one key at a time: dsh.Usage exposes input tokens under the alias
    # "in", which is a Python keyword and cannot be passed as dict(in=...).
    summary["llm_calls"] = usage.get("calls")
    summary["input_tokens"] = usage.get("input_tokens")
    summary["output_tokens"] = usage.get("output_tokens")
    summary["cache_read_tokens"] = usage.get("cache_read_tokens")
    summary["cache_write_tokens"] = usage.get("cache_write_tokens")
    summary["reasoning_tokens"] = usage.get("reasoning_tokens")
    summary["total_tokens"] = usage.get("total_tokens")
    summary["turns"] = (result.get("stats") or {}).get("turns")
    summary["tool_calls"] = (result.get("stats") or {}).get("tool_calls")
    summary["records"] = (result.get("stats") or {}).get("records")
    summary["folds"] = (result.get("stats") or {}).get("folds")
    summary["action_counts"] = result.get("action_counts")
    summary["final_score"] = result.get("final_score")
    summary["error"] = result.get("error")
    summary["out"] = str(args.out)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
