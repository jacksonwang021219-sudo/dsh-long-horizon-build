"""Single-turn UltraHorizon episode: the environment's actions are DSH tools.

This is the faithful mapping of the benchmark's own loop. The episode is ONE
user turn: the agent explores through tool calls (`input_sequences`, `move`,
`commit_final_result`, …), the environment's observations come back as tool
results, and the turn ends only when the agent stops acting — exactly the ReAct
shape the benchmark was built around.

The environment lives in this process; a small HTTP bridge serves its action
list and forwards calls, and `uh_bridge_plugin.mjs` exposes those actions as
DSH tools inside the session's preset.

    python3 uh_tools.py --env seq --steps 20 --seed 7 --difficulty easy \
        --preset standard --base-url http://127.0.0.1:PORT --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import Harness, stats_from_events  # noqa: E402
from ultrahorizon_driver import (  # noqa: E402
    Action,
    EnvBridge,
    _import_envs,
    load_api_key,
    patch_placeholder_judge_config,
    reroute_judge_client,
    run_judge,
    seed_scope,
    write_result,
)

_import_envs()  # put bench/UltraHorizon on sys.path before importing envs.common

PRESET_ROOT = Path.home() / ".dsh" / ".agent-presets"


def tool_schemas(bridge: EnvBridge) -> list[dict[str, Any]]:
    """The environment's actions as tool definitions the bridge plugin registers."""
    schemas: list[dict[str, Any]] = []

    def add(name: str, description: str, properties: dict[str, Any], required: list[str]) -> None:
        schemas.append({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                **({"required": required} if required else {}),
            },
        })

    names = bridge.action_names()
    for name in names:
        if name == "input_sequences":
            add(name, "Submit a pair of 5-character sequences (alphabet A-E, at least two distinct letters each) and receive the transformation chain for this step.",
                {"main_sequence": {"type": "string", "description": "The main 5-character sequence."},
                 "vice_sequence": {"type": "string", "description": "The vice 5-character sequence."}},
                ["main_sequence", "vice_sequence"])
        elif name == "get_game_info":
            add(name, "Read the game rules, the current step counter and what a commit expects.", {}, [])
        elif name == "get_history":
            add(name, "Read back recent inputs and outputs from this episode.",
                {"last_n": {"type": "integer", "description": "How many recent entries to return."}}, [])
        elif name == "move":
            add(name, "Move one tile in the grid and receive the resulting state.",
                {"direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Direction to move."}},
                ["direction"])
        elif name == "get_current_state":
            add(name, "Read the local 3x3 view, energy, score and step counter.", {}, [])
        elif name == "get_full_map":
            add(name, "Read the whole map with every tile visited so far.", {}, [])
        elif name == "reset":
            add(name, "Reset the grid to its initial layout and restore energy.", {}, [])
        elif name == "conduct_cross":
            add(name, "Cross two organisms and receive their offspring.",
                {"parent1_id": {"type": "integer", "description": "First parent organism id."},
                 "parent2_id": {"type": "integer", "description": "Second parent organism id."},
                 "num_offspring": {"type": "integer", "description": "Number of offspring to produce."}},
                ["parent1_id", "parent2_id"])
        elif name == "query_organisms":
            add(name, "Read stored organisms by id range.",
                {"start_id": {"type": "integer", "description": "First id to return."},
                 "end_id": {"type": "integer", "description": "Last id to return."}}, ["start_id"])
        elif name == "remove_organisms":
            add(name, "Free lab space by removing organisms.",
                {"organism_ids": {"type": "array", "items": {"type": "integer"},
                                  "description": "Ids to remove."}}, ["organism_ids"])
        elif name == "get_lab_status":
            add(name, "Read lab capacity, organism count and experiment counter.", {}, [])
        elif name in {"commit", "finish", "submit"}:
            add("commit_final_result",
                "Submit your final answer once you have enough evidence. Score it gets is final: this also ends the episode.",
                {"content": {"type": "string", "description": "Your complete answer: the discovered rules with the evidence for each."}},
                ["content"])
    return schemas


class Bridge:
    """Owns the environment and serves it over HTTP to the Cordis plugin."""

    def __init__(self, env: EnvBridge) -> None:
        self.env = env
        self.calls: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port = 0

    def start(self) -> int:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
                if self.path.startswith("/tools"):
                    self._reply({"actions": tool_schemas(bridge.env)})
                elif self.path.startswith("/progress"):
                    self._reply(bridge.env.progress())
                else:
                    self._reply({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    request = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as error:
                    self._reply({"isError": True, "error": f"bad request body: {error}"})
                    return
                name = str(request.get("name"))
                args = request.get("args") if isinstance(request.get("args"), dict) else {}
                if name in {"commit", "commit_final_result", "finish", "submit"}:
                    # EnvBridge routes the "commit" alias to commit_final_result.
                    args = {"content": args.get("content", "")}
                    name = "commit"
                action = Action(name=name, args=args)
                try:
                    observation = bridge.env.call(action)
                    is_error = isinstance(observation, dict) and observation.get("success") is False
                except Exception as error:  # noqa: BLE001 -- surfaced to the model as a tool error
                    bridge.calls.append({"name": name, "args": args, "error": f"{type(error).__name__}: {error}"})
                    self._reply({"isError": True, "error": f"{type(error).__name__}: {error}"})
                    return
                bridge.calls.append({"name": name, "args": args, "error": None if not is_error else "rejected"})
                self._reply({"observation": observation, "isError": bool(is_error)})

            def log_message(self, *_args: Any) -> None:  # silence per-request logging
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.port

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


def write_bridge_presets(base_preset: str, bridge_plugin: Path, url: str) -> str:
    """Copy one shipped preset and add the bridge row; return the new preset id."""
    roots = [
        Path(__file__).resolve().parents[2] / "deepseek-harness" / "apps" / "cli" / "config" / "agent-presets",
        PRESET_ROOT,
    ]
    source = next((root / base_preset for root in roots if (root / base_preset).exists()), None)
    if source is None:
        raise SystemExit(f"preset {base_preset!r} not found in {', '.join(str(root) for root in roots)}")
    preset_id = f"uh-{base_preset}"
    target = PRESET_ROOT / preset_id
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_dir():
            import shutil
            shutil.copytree(item, target / item.name, dirs_exist_ok=True)
        else:
            target.joinpath(item.name).write_text(item.read_text())
    composition = (source / "agent.cordis.yml").read_text().rstrip("\n")
    composition += f"""

# ── UltraHorizon environment tools (benchmark bridge) ───────────────────────

- id: uh-bridge
  name: '{bridge_plugin}'
  config:
    url: '{url}'
"""
    (target / "agent.cordis.yml").write_text(composition)
    (target / "preset.yml").write_text(
        f"name: {base_preset} + UltraHorizon 环境工具\n"
        "description: 标准/长程模式的副本，额外挂载一个把 UltraHorizon 环境动作暴露成 DSH 工具的桥接插件（评测用）。\n"
    )
    return preset_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["seq", "grid", "bio"], required=True)
    parser.add_argument("--steps", type=int, default=20, help="required environment steps (the env's commit gate)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--preset", default="standard", help="shipped preset the episode runs under")
    parser.add_argument("--base-url", default="http://127.0.0.1:58422")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--turn-timeout", type=float, default=3600.0, help="wall clock cap for the single turn")
    parser.add_argument("--max-tool-calls", type=int, default=400, help="abort the turn past this many environment calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    envs = _import_envs()
    env_cls = envs[args.env]
    from envs.common import Difficulty  # noqa: PLC0415 -- the benchmark's own enum

    difficulty = Difficulty[args.difficulty.upper()]
    env_module = sys.modules[env_cls.__module__]
    api_key = load_api_key()

    result: dict[str, Any] = {
        "env": args.env,
        "seed": args.seed,
        "preset": args.preset,
        "difficulty": args.difficulty,
        "step_cap": args.steps,
        "mapping": "single user turn; environment actions exposed as DSH tools",
        "base_url": args.base_url,
        "notes": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    restore_seed = seed_scope(args.seed)
    undo_judge_config = patch_placeholder_judge_config(env_cls, api_key, args.judge_base_url, args.judge_model)
    result["notes"].append(
        "random.seed pinned for the whole episode; load_judge_config replaced on the imported class "
        "(both reversible; the cloned repo is never modified)"
    )
    try:
        env = env_cls(difficulty=difficulty, required_steps=args.steps)
        if args.judge:
            undo_judge = reroute_judge_client(env_module, api_key, args.judge_base_url, args.judge_model)
            result["notes"].append(
                f"commit_final_result judges through {args.judge_base_url} / {args.judge_model} (reversible)"
            )
        else:
            undo_judge = None

        bridge = EnvBridge(env, args.env, verbose=args.verbose)
        server = Bridge(bridge)
        port = server.start()
        url = f"http://127.0.0.1:{port}"
        result["bridge_port"] = port

        preset_id = write_bridge_presets(args.preset, Path(__file__).resolve().parent / "uh_bridge_plugin.mjs", url)
        result["bridge_preset"] = preset_id

        harness = Harness(args.base_url)
        cwd = Path(__file__).resolve().parents[1] / "runs" / "ultrahorizon" / f"tools-{args.env}-seed{args.seed}-{args.preset}"
        cwd.mkdir(parents=True, exist_ok=True)
        session_id = harness.create_session(str(cwd), preset_id)
        result["session_id"] = session_id
        result["cwd"] = str(cwd)

        first_message = (
            f"{env.env_prompt}\n\n---\n"
            "You are working inside one task. Use the environment tools to run experiments, and reason from "
            "their results. The environment accepts a commit once the required number of steps has been taken; "
            "when you can state the hidden rules with the evidence for each, call `commit_final_result` with your "
            "complete answer. The commit is mandatory: never end your reply without having called "
            "`commit_final_result` at least once, and if it is rejected, adjust your answer and call it again."
        )
        started = time.time()
        stop_watchdog = threading.Event()

        def watchdog() -> None:
            """Cancel the episode when it exceeds its tool-call or wall-clock budget."""
            deadline = started + args.turn_timeout
            while not stop_watchdog.wait(5.0):
                if len(server.calls) > args.max_tool_calls:
                    result["notes"].append(
                        f"aborted: environment calls passed {args.max_tool_calls}"
                    )
                elif time.time() > deadline:
                    result["notes"].append(f"aborted: wall clock passed {args.turn_timeout:.0f}s")
                else:
                    continue
                try:
                    harness.cancel(session_id)
                except Exception as error:  # noqa: BLE001 -- recorded, not fatal
                    result["notes"].append(f"cancel failed: {type(error).__name__}: {error}")
                return

        threading.Thread(target=watchdog, daemon=True).start()
        harness.prompt(session_id, first_message)
        idle = harness.wait_idle(session_id, timeout=args.turn_timeout + 120)
        stop_watchdog.set()
        result["wall_seconds"] = round(time.time() - started, 1)
        result["idle"] = idle
        stats = stats_from_events(harness.read_log(session_id), session_id)
        result["stats"] = stats.as_dict()
        result["env_calls"] = [
            {"name": call["name"], "error": call["error"]} for call in server.calls
        ]
        result["env_counter"] = bridge.progress()
        result["committed"] = bridge.is_done()
        result["finished"] = bool(bridge.is_done())
        result["agent_final_answer"] = stats.last_assistant_text

        final = getattr(env, "final_result", None)
        if isinstance(final, dict):
            result["final_result"] = final
            judge_result = final.get("judge_result") or {}
            result["final_score"] = judge_result.get("final_score")
            result["score_breakdown"] = judge_result.get("score_breakdown")
        if result.get("final_score") is None and args.judge:
            try:
                judge = run_judge(env, args.env, env_cls, str(stats.last_assistant_text),
                                  api_key, args.judge_base_url, args.judge_model)
                result["host_judge"] = judge
                result["final_score"] = judge.get("final_score")
                result["score_breakdown"] = judge.get("score_breakdown")
            except Exception as error:  # noqa: BLE001 -- recorded, not fatal
                result["notes"].append(f"host judge failed: {type(error).__name__}: {error}")

        server.stop()
        if undo_judge is not None:
            undo_judge()
        result["random_seed_restored"] = False
    finally:
        restore_seed()
        undo_judge_config()
        result["random_seed_restored"] = True
        result["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_result(args.out, result)

    print(json.dumps({
        "env": result["env"], "preset": result["preset"], "score": result.get("final_score"),
        "committed": result.get("committed"), "env_steps": (result.get("env_counter") or {}).get("value"),
        "required_steps": args.steps, "outer_turns": (result.get("stats") or {}).get("turns"),
        "steps": (result.get("stats") or {}).get("steps"),
        "tokens": ((result.get("stats") or {}).get("usage") or {}).get("total_tokens"),
        "wall_seconds": result.get("wall_seconds"), "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
