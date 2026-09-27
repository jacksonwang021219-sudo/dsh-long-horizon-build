"""Price and summarise the AgencyBench arms: score, tokens, and money.

Every input token a run spent is billed either at the cache-hit price or at the
cache-miss price, and the split is the whole difference between the arms, so the
two are summed and priced separately:

    cost = miss_tokens * miss_price + cache_read_tokens * hit_price
         + output_tokens * output_price

`input_tokens` is what the provider charged at the miss price (the prompt minus
whatever it served from cache), `cache_read_tokens` is what it served from
cache, and `output_tokens` already contains reasoning. The note keeper's own
calls are priced the same way from the session's `recorder-usage.jsonl`, because
they never enter the session log.

    python3 analyze_v1v2.py                       # every runs/dsh-{v1,v2}-r*.json
    python3 analyze_v1v2.py --extra label=path    # plus older arms, labelled
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# DeepSeek-V4.1-Flash list price, CNY per million tokens. Two windows: the
# off-peak price applies outside 09:00-12:00 and 14:00-18:00 Beijing time on
# weekdays, and 2026-09-27 is a Sunday, so every run here is off-peak.
PRICES = {
    "off-peak": {"hit": 0.02, "miss": 1.0, "output": 4.0},
    "peak": {"hit": 0.04, "miss": 2.0, "output": 8.0},
}


def money(tokens: dict[str, int], window: str) -> float:
    """Price one token bucket under one price window, in CNY."""
    price = PRICES[window]
    return (
        tokens.get("input_tokens", 0) * price["miss"]
        + tokens.get("cache_read_tokens", 0) * price["hit"]
        + tokens.get("output_tokens", 0) * price["output"]
    ) / 1_000_000


def load(path: Path, label: str) -> dict[str, object]:
    """Reduce one run report to the row the summary prints."""
    report = json.loads(path.read_text())
    session = report.get("session") or {}
    usage = session.get("usage") or {}
    recorder = report.get("recorder") or {}
    leakage = report.get("leakage") or {}
    turns = report.get("turns") or []
    pass1 = (report.get("pass1") or {}).get("total")
    pass2 = (report.get("pass2") or {}).get("total")
    parent = {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    keeper = {
        "input_tokens": recorder.get("input_tokens", 0),
        "cache_read_tokens": recorder.get("cache_read_tokens", 0),
        "output_tokens": recorder.get("output_tokens", 0),
    }
    total = {key: parent[key] + keeper[key] for key in parent}
    return {
        "label": label,
        "path": str(path),
        "mode": report.get("mode"),
        "preset": report.get("preset"),
        "session_id": report.get("session_id"),
        "pass1": pass1,
        "pass2": pass2,
        "turns": session.get("turns"),
        "steps": session.get("steps"),
        "tool_calls": session.get("tool_calls"),
        "folds": session.get("folds"),
        "seconds": round(sum(float(turn.get("seconds") or 0) for turn in turns)),
        "attempts": sum(int(turn.get("attempts") or 0) for turn in turns),
        "parent_tokens": parent,
        "recorder": {**keeper, "calls": recorder.get("calls", 0), "failed_calls": recorder.get("failed_calls", 0)},
        "all_tokens": total,
        "cost_off_peak": money(parent, "off-peak") + money(keeper, "off-peak"),
        "cost_peak": money(parent, "peak") + money(keeper, "peak"),
        "recorder_cost_off_peak": money(keeper, "off-peak"),
        "cache_hit_share": (
            parent["cache_read_tokens"] / (parent["cache_read_tokens"] + parent["input_tokens"])
            if parent["cache_read_tokens"] + parent["input_tokens"] else 0.0
        ),
        "leak_markers": leakage.get("markers", {}),
        # The tight signal: a call naming the grader's source or a rubric point
        # read the rubric. `AgencyBench` alone can be the agent echoing its own
        # cwd, so it is kept separate.
        "rubric_marker_calls": sum(
            (leakage.get("markers") or {}).get(name, 0)
            for name in ("eval_task.py", "Score_", "failed_points", "evalspace")
        ) if leakage else None,
        "leak_marker_calls": leakage.get("marker_calls"),
        "paths_outside_root_calls": leakage.get("paths_outside_root_calls"),
        "leak_examples": leakage.get("examples", []),
        "agent_root": report.get("agent_root") or report.get("run_root"),
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group rows by arm and reduce each group to mean and range."""
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["mode"]), []).append(row)
    out = []
    for mode, items in groups.items():
        finals = [item["pass2"] if item["pass2"] is not None else item["pass1"] for item in items]
        numbers = [value for value in finals if isinstance(value, (int, float))]
        out.append({
            "mode": mode,
            "n": len(items),
            "finals": finals,
            "mean": statistics.mean(numbers) if numbers else None,
            "min": min(numbers) if numbers else None,
            "max": max(numbers) if numbers else None,
            "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
            "mean_cost": statistics.mean([float(item["cost_off_peak"]) for item in items]),
            "mean_steps": statistics.mean([int(item["steps"] or 0) for item in items]),
            "mean_seconds": statistics.mean([int(item["seconds"] or 0) for item in items]),
            "mean_cache_hit_share": statistics.mean([float(item["cache_hit_share"]) for item in items]),
            "mean_recorder_calls": statistics.mean([int(item["recorder"]["calls"]) for item in items]),
            "mean_recorder_cost": statistics.mean([float(item["recorder_cost_off_peak"]) for item in items]),
            "rubric_marker_calls": [item["rubric_marker_calls"] for item in items],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", action="append", default=[],
                        help="label=path of another run report to include")
    parser.add_argument("--glob", default="runs/dsh-*-r*.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(str(ROOT / args.glob)))
    rows = [load(path, path.stem) for path in paths]
    for entry in args.extra:
        label, _, path = entry.partition("=")
        rows.append(load(Path(path), label))

    summary = summarise(rows)
    payload = {"rows": rows, "summary": summary}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    header = f"{'arm':<12} {'n':>2} {'final':>8} {'mean':>5} {'steps':>6} {'min':>6} {'¥(off)':>7} {'hit%':>5} {'rec':>4} {'¥rec':>6} {'leak':>5}"
    print(header)
    for item in summary:
        finals = ",".join(str(value) for value in item["finals"])
        leak = ",".join(str(value) for value in item["rubric_marker_calls"])
        print(
            f"{item['mode']:<12} {item['n']:>2} {finals:>8} {item['mean']:>5.1f} "
            f"{item['mean_steps']:>6.0f} {item['mean_seconds'] / 60:>5.0f}m "
            f"{item['mean_cost']:>7.2f} {item['mean_cache_hit_share'] * 100:>4.0f}% "
            f"{item['mean_recorder_calls']:>4.1f} {item['mean_recorder_cost']:>6.3f} {leak:>5}"
        )
    print()
    for row in rows:
        print(
            f"{row['label']:<14} pass1={row['pass1']} pass2={row['pass2']} steps={row['steps']} "
            f"folds={row['folds']} turns={row['turns']} attempts={row['attempts']} "
            f"parent(miss={row['parent_tokens']['input_tokens']}, hit={row['parent_tokens']['cache_read_tokens']}, "
            f"out={row['parent_tokens']['output_tokens']}) "
            f"recorder(calls={row['recorder']['calls']}, failed={row['recorder']['failed_calls']}, "
            f"miss={row['recorder']['input_tokens']}, hit={row['recorder']['cache_read_tokens']}, "
            f"out={row['recorder']['output_tokens']}) "
            f"¥{row['cost_off_peak']:.3f} leak={row['rubric_marker_calls']} outside={row['paths_outside_root_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
