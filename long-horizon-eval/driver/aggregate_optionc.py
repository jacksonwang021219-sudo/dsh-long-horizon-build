"""Aggregate the option-C batch: nine scenarios × two presets.

Every scenario family reports on the same 0–10 scale — the rubric family as a
per-subtask `best_score`, the loss/recall families as a scenario-level
`final_score` — so a scenario's score is its mean over the subtasks it defines.

    python3 aggregate_optionc.py --batch ../runs/optionC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench" / "AgencyBench" / "AgencyBench-v2"

# slug -> (scenario path, subtask count, family)
SCENARIOS: dict[str, tuple[str, int, str]] = {
    "be1": ("Backend/scenario1", 5, "rubric"),
    "be3": ("Backend/scenario3", 5, "rubric"),
    "co1": ("Code/scenario1", 5, "rubric"),
    "co6": ("Code/scenario6", 1, "loss"),
    "co7": ("Code/scenario7", 1, "loss"),
    "re2": ("Research/scenario2", 5, "rubric"),
    "re3": ("Research/scenario3", 5, "rubric"),
    "re4": ("Research/scenario4", 2, "recall"),
    "re5": ("Research/scenario5", 2, "recall"),
}


def scenario_score(slug: str, model_name: str) -> dict[str, object]:
    """Read one scenario's grader output and reduce it to a 0–10 score.

    Three grader families, all on a 0–10 scale: per-subtask rubric scores, a
    scenario-level loss score, and a scenario-level normalized recall. The recall
    family needs a judge the scenario hardcodes to an internal gateway model; when
    that judge cannot run here, the comparable figure is the mean of its
    per-subtask rubric scores, exactly as the rule-scored family reports.
    """
    scenario, subtasks, family = SCENARIOS[slug]
    meta_path = BENCH / scenario / model_name / "meta_eval.json"
    if not meta_path.exists():
        return {"score": None, "detail": "no meta_eval.json", "subtasks": subtasks}
    meta = json.loads(meta_path.read_text())
    rows = meta.get("subtasks") or []

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    if family == "rubric":
        per = [row.get("best_score") for row in rows if isinstance(row.get("best_score"), (int, float))]
        return {"score": mean(per), "per_subtask": per, "score_basis": "per-subtask rubric", "subtasks": subtasks}
    if family == "loss":
        return {"score": float(meta.get("final_score") or 0),
                "raw": {"final_score": meta.get("final_score"), "best_loss": meta.get("best_loss")},
                "subtasks": subtasks}
    # Both arms are scored by the same rule: the mean of the per-subtask rubric
    # scores. The scenario's own normalized recall is recorded as supplementary
    # evidence, never as the comparison's score — its judge is hardcoded to an
    # internal gateway model and had to be substituted here.
    per = [row.get("final_score") for row in rows if isinstance(row.get("final_score"), (int, float))]
    return {"score": mean(per), "per_subtask": per,
            "score_basis": "per-subtask rubric; scenario recall supplementary",
            "raw": {"final_score": meta.get("final_score"), "rubric_recall": meta.get("rubric_recall")},
            "subtasks": subtasks}


def run_totals(model_name: str, batch: Path) -> dict[str, object]:
    """Token, step and wall-clock totals for one run, from the runner's report."""
    report_path = batch / f"{model_name}.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    turns = report.get("turns") or []
    last = turns[-1] if turns else {}
    usage = last.get("usage") or {}
    return {
        "tokens": usage.get("total_tokens"),
        "fresh": usage.get("input_tokens"),
        "cache": usage.get("cache_read_tokens"),
        "output": usage.get("output_tokens"),
        "steps": last.get("steps"),
        "records": last.get("records"),
        "folds": last.get("folds"),
        "seconds": round(sum(float(turn.get("seconds", 0) or 0) for turn in turns), 1),
        "turns_used": len(turns),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=str(Path(__file__).resolve().parent.parent / "runs" / "optionC"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    batch = Path(args.batch)

    rows: list[dict[str, object]] = []
    for slug, (scenario, subtasks, family) in SCENARIOS.items():
        for preset, prefix in (("standard", "std"), ("long-horizon", "lh")):
            model_name = f"{prefix}-{slug}"
            rows.append({
                "slug": slug, "scenario": scenario, "preset": preset, "family": family,
                "model_name": model_name, "subtasks": subtasks,
                **(scenario_score(slug, model_name) or {}),
                **run_totals(model_name, batch),
            })

    def table(selected: list[dict[str, object]]) -> str:
        lines = ["| scenario | 子任务 | 标准模式 | 长程模式 | 差 |", "| --- | --- | --- | --- | --- |"]
        for slug, (scenario, subtasks, _family) in SCENARIOS.items():
            pair = {row["preset"]: row for row in selected if row["slug"] == slug}
            std = pair.get("standard", {}).get("score")
            lh = pair.get("long-horizon", {}).get("score")
            delta = None if std is None or lh is None else round(float(lh) - float(std), 2)
            lines.append(f"| {scenario} | {subtasks} | {std} | {lh} | {delta} |")
        return "\n".join(lines)

    def tokens_table(selected: list[dict[str, object]]) -> str:
        lines = ["| scenario | 标准 token | 长程 token | 长程/标准 | 标准 步 | 长程 步 | 标准 墙钟 | 长程 墙钟 |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for slug, (scenario, _subtasks, _family) in SCENARIOS.items():
            pair = {row["preset"]: row for row in selected if row["slug"] == slug}
            std, lh = pair.get("standard", {}), pair.get("long-horizon", {})
            ratio = None
            if std.get("tokens") and lh.get("tokens"):
                ratio = round(float(lh["tokens"]) / float(std["tokens"]), 2)
            def fmt(value: object, scale: float = 1.0) -> str:
                return f"{float(value) / scale:,.0f}" if isinstance(value, (int, float)) else "—"
            lines.append(
                f"| {scenario} | {fmt(std.get('tokens'), 1e6)} M | {fmt(lh.get('tokens'), 1e6)} M | {ratio} | "
                f"{std.get('steps', '—')} | {lh.get('steps', '—')} | {fmt(std.get('seconds'), 60)} min | "
                f"{fmt(lh.get('seconds'), 60)} min |"
            )
        return "\n".join(lines)

    complete = [row for row in rows if row.get("score") is not None and row.get("tokens")]
    print("## 分数（0–10，scenario 内均值）\n")
    print(table(rows))
    print("\n## token 与耗时\n")
    print(tokens_table(rows))

    summary: dict[str, object] = {"rows": rows, "complete": len(complete)}
    for preset in ("standard", "long-horizon"):
        scored = [row for row in complete if row["preset"] == preset]
        if not scored:
            continue
        summary[preset] = {
            "scenarios": len(scored),
            "mean_score": round(sum(float(row["score"]) for row in scored) / len(scored), 3),
            "total_tokens": sum(int(row["tokens"]) for row in scored),
            "total_steps": sum(int(row["steps"] or 0) for row in scored),
            "total_seconds": round(sum(float(row["seconds"] or 0) for row in scored), 1),
            "records": sum(int(row.get("records") or 0) for row in scored),
            "folds": sum(int(row.get("folds") or 0) for row in scored),
        }
    print("\n## 汇总\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
