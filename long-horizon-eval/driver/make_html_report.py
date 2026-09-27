"""Generate the standalone AgencyBench evaluation report (HTML, no dependencies).

Reads the run artifacts under `runs/` and writes one self-contained page with
inline CSS and inline SVG charts — openable offline, no scripts, no CDN.

    python3 make_html_report.py [--out ../report/agencybench-report.html]
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_DEFAULT = ROOT / "report" / "agencybench-report.html"

# ---------------------------------------------------------------- statistics


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    maxit, eps, fpmin = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1 - x) / b


def t_p_value(t: float, df: int) -> float:
    """Two-sided p-value of a t statistic."""
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(p: float, df: int) -> float:
    """Inverse t CDF by bisection."""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - betai(df / 2.0, 0.5, df / (df + mid * mid)) / 2 < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value."""
    if n == 0:
        return 1.0

    def pmf(i: int) -> float:
        return math.comb(n, i) * 0.5 ** n

    observed = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12))


def wilcoxon_exact(values: list[float]) -> tuple[float, float, int]:
    """Exact two-sided Wilcoxon signed-rank test (zero differences dropped)."""
    nonzero = [value for value in values if value != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0
    order = sorted(range(n), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        average = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    total = n * (n + 1) / 2
    observed = min(w_plus, total - w_plus)
    hits = 0
    for mask in range(1 << n):
        plus = sum(ranks[k] for k in range(n) if mask >> k & 1)
        if min(plus, total - plus) <= observed + 1e-9:
            hits += 1
    return w_plus, hits / (1 << n), n


def paired_report(differences: list[float], label: str, ratio: bool = False) -> dict[str, object]:
    """Paired t-test, sign test, and confidence intervals for one comparison."""
    n = len(differences)
    m = mean(differences)
    sd = stdev(differences)
    se = sd / math.sqrt(n)
    t = m / se
    p = t_p_value(t, n - 1)
    intervals = {}
    for confidence in (0.80, 0.85, 0.90, 0.95):
        crit = t_quantile(1 - (1 - confidence) / 2, n - 1)
        low, high = m - crit * se, m + crit * se
        intervals[confidence] = (
            (math.exp(low), math.exp(high)) if ratio else (low, high)
        )
    wins = sum(1 for d in differences if d > 0)
    losses = sum(1 for d in differences if d < 0)
    w_plus, wilcoxon_p, wilcoxon_n = wilcoxon_exact(differences)
    return {
        "w_plus": w_plus, "wilcoxon_p": wilcoxon_p, "wilcoxon_n": wilcoxon_n,
        "label": label, "n": n, "mean": m, "sd": sd, "se": se, "t": t,
        "df": n - 1, "p": p, "cohens_d": m / sd, "intervals": intervals,
        "wins": wins, "losses": losses, "ties": n - wins - losses,
        "sign_p": binom_two_sided(wins, wins + losses) if wins + losses else 1.0,
        "ratio": ratio,
    }


# ------------------------------------------------------------------- charting


def bar_chart(rows: list[tuple[str, float, float]], width: int = 900, height: int = 320,
              y_max: float | None = None, suffix: str = "") -> str:
    """Two-series grouped horizontal bar chart as inline SVG."""
    label_w, pad = 250, 60
    plot_w = width - label_w - pad
    bar_h, gap = 16, 10
    n = len(rows)
    height = max(height, n * (bar_h * 2 + gap) + 70)
    top = 50
    y_max = y_max or max(max(a, b) for _, a, b in rows) * 1.15 or 1
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')
    parts.append(f'<text x="{label_w - 10}" y="24" text-anchor="end" font-size="13" fill="#6b7280">'
                 f'标准模式</text>')
    parts.append(f'<text x="{label_w - 10}" y="42" text-anchor="end" font-size="13" fill="#0f766e">'
                 f'长程任务模式</text>')
    for tick in range(5):
        value = y_max * tick / 4
        x = label_w + plot_w * tick / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 24}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-size="11" '
                     f'fill="#9ca3af">{value:.1f}{suffix}</text>')
    for index, (label, a, b) in enumerate(rows):
        y = top + index * (bar_h * 2 + gap)
        parts.append(f'<text x="{label_w - 14}" y="{y + 26}" text-anchor="end" font-size="12" '
                     f'fill="#374151">{html.escape(label)}</text>')
        for value, color, offset in ((a, "#9ca3af", 0), (b, "#0f766e", bar_h + 2)):
            w = max(1.0, plot_w * min(value, y_max) / y_max)
            parts.append(f'<rect x="{label_w}" y="{y + offset}" width="{w:.1f}" height="{bar_h}" '
                         f'rx="3" fill="{color}"/>')
            parts.append(f'<text x="{label_w + w + 6:.1f}" y="{y + offset + 12}" font-size="11" '
                         f'fill="#374151">{value:.2f}{suffix}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def line_chart(series: list[tuple[str, list[float], str]], labels: list[str],
               width: int = 900, height: int = 300, suffix: str = "") -> str:
    """Cumulative-token line chart as inline SVG."""
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    y_max = max(max(values) for _, values, _ in series) * 1.1
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    parts.append('<rect width="100%" height="100%" fill="#fff"/>')
    for tick in range(5):
        y = pad_t + plot_h - plot_h * tick / 4
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="#9ca3af">{y_max * tick / 4 / 1e6:.0f}M</text>')
    count = len(labels)
    for index, label in enumerate(labels):
        x = pad_l + (plot_w * index / max(1, count - 1))
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" font-size="11" '
                     f'fill="#6b7280">{html.escape(label)}</text>')
    for name, values, color in series:
        points = []
        for index, value in enumerate(values):
            x = pad_l + (plot_w * index / max(1, count - 1))
            y = pad_t + plot_h - plot_h * value / y_max
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        for point in points:
            x, y = point.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
    legend_x = pad_l
    for name, _values, color in series:
        parts.append(f'<rect x="{legend_x}" y="8" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="18" font-size="12" fill="#374151">{html.escape(name)}</text>')
        legend_x += 150
    parts.append('</svg>')
    return "\n".join(parts)


# ------------------------------------------------------------------ assembling


def load_rounds() -> list[dict[str, object]]:
    """The four single-scenario rounds on Backend/scenario1."""
    rounds = [
        ("standard", "第 1 轮", "full-standard.json"),
        ("standard", "第 2 轮", "full-standard-r2.json"),
        ("long-horizon", "第 1 轮", "full-long-horizon.json"),
        ("long-horizon", "第 2 轮", "full-long-horizon-fixed.json"),
    ]
    out = []
    for preset, label, name in rounds:
        path = RUNS / name
        if not path.exists():
            continue
        report = json.loads(path.read_text())

        def scores(key: str) -> list[float]:
            rows = (report.get(key) or {}).get("subtasks") or []
            return [row["best_score"] for row in rows if isinstance(row.get("best_score"), (int, float))]

        first, second = scores("pass1"), scores("pass2")
        merged = [max(a, b) for a, b in zip(first, second)] if second else first
        last = report["turns"][-1]
        out.append({
            "preset": preset, "label": label, "first": first, "second": second, "merged": merged,
            "pass1": sum(first), "pass2": sum(merged),
            "tokens": last["usage"]["total_tokens"], "steps": last["steps"],
            "records": last.get("records", 0), "folds": last.get("folds", 0),
            "seconds": sum(float(turn.get("seconds", 0) or 0) for turn in report["turns"]),
        })
    return out


def fmt_m(value: float | None) -> str:
    return "—" if value is None else f"{value / 1e6:,.1f} M"


def esc(value: object) -> str:
    return html.escape(str(value))


def build() -> str:
    """Assemble the page: results first, then how to reproduce them."""
    data = json.loads((RUNS / "optionC" / "optionC-final.json").read_text())
    rows = data["rows"]
    summary = {k: v for k, v in data.items() if k != "rows"}
    pairs: dict[str, dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(row["scenario"], {})[row["preset"]] = row
    order = [row["scenario"] for row in rows if row["preset"] == "standard"]
    std, lh = summary["standard"], summary["long-horizon"]

    score_rows = [(n, pairs[n]["standard"]["score"], pairs[n]["long-horizon"]["score"]) for n in order]
    token_rows = [(n, 1.0, pairs[n]["long-horizon"]["tokens"] / pairs[n]["standard"]["tokens"]) for n in order]
    step_rows = [(n, pairs[n]["standard"]["tokens"] / pairs[n]["standard"]["steps"],
                  pairs[n]["long-horizon"]["tokens"] / pairs[n]["long-horizon"]["steps"]) for n in order]

    score_stats = paired_report([lh_ - s_ for _n, s_, lh_ in score_rows], "分数差")
    token_stats = paired_report([math.log(pairs[n]["long-horizon"]["tokens"] / pairs[n]["standard"]["tokens"])
                                 for n in order], "token 比", ratio=True)

    rounds = load_rounds()
    rounds_chart = line_chart(
        [("标准模式", [r["tokens"] for r in rounds if r["preset"] == "standard"], "#9ca3af"),
         ("长程任务模式", [r["tokens"] for r in rounds if r["preset"] == "long-horizon"], "#0f766e")],
        ["第 1 轮", "第 2 轮"],
    )
    score_chart = bar_chart(score_rows, y_max=11)
    ratio_chart = bar_chart(token_rows, y_max=2.0, suffix="×")
    step_chart = bar_chart([(n, a / 1000, b / 1000) for n, a, b in step_rows],
                           y_max=max(b / 1000 for _n, _a, b in step_rows) * 1.15, suffix="K")

    def delta_class(value: float) -> str:
        return "win" if value > 0 else ("tie" if value == 0 else "loss")

    def score_cell(value: float) -> str:
        return f"{value:.2f}"

    def main_rows() -> str:
        out = []
        for name in order:
            s, l = pairs[name]["standard"], pairs[name]["long-horizon"]
            delta = l["score"] - s["score"]
            ratio = l["tokens"] / s["tokens"]
            out.append(
                f"<tr><td class='l'>{esc(name)}</td><td>{s['subtasks']}</td>"
                f"<td>{score_cell(s['score'])}</td><td>{score_cell(l['score'])}</td>"
                f"<td class='{delta_class(delta)}'>{delta:+.2f}</td>"
                f"<td>{fmt_m(s['tokens'])}</td><td>{fmt_m(l['tokens'])}</td>"
                f"<td class='{delta_class(1 - ratio)}'>{ratio:.2f}×</td>"
                f"<td>{s['seconds'] / 60:.0f} / {l['seconds'] / 60:.0f}</td></tr>")
        return "\n".join(out)

    def round_rows() -> str:
        out = []
        for r in rounds:
            name = "标准模式" if r["preset"] == "standard" else "长程任务模式"
            scores = ", ".join(f"{int(x)}" if float(x).is_integer() else f"{x:.2f}" for x in r["merged"])
            out.append(
                f"<tr><td class='l'>{name}</td><td>{esc(r['label'])}</td><td>{r['pass1']}</td>"
                f"<td class='win'>{r['pass2']}</td><td class='l'>{scores}</td>"
                f"<td>{fmt_m(r['tokens'])}</td><td>{r['steps']}</td>"
                f"<td>{r['tokens'] / max(1, r['steps']):,.0f}</td>"
                f"<td>{r['records']}/{r['folds']}</td><td>{r['seconds'] / 60:.0f}</td></tr>")
        return "\n".join(out)

    def interval_rows(stats: dict) -> str:
        out = []
        for confidence, (low, high) in stats["intervals"].items():
            significant = (low > 1.0 or high < 1.0) if stats["ratio"] else not (low <= 0 <= high)
            shown = f"{low:.2f}× ~ {high:.2f}×" if stats["ratio"] else f"{low:+.2f} ~ {high:+.2f}"
            out.append(f"<tr><td>{confidence:.0%}</td><td>{shown}</td>"
                       f"<td class='{'win' if significant else 'loss'}'>{'显著' if significant else '不显著'}</td></tr>")
        return "\n".join(out)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgencyBench 评测：标准模式与长程任务模式</title>
<style>
  body {{ margin:0; background:#f7f8fa; color:#16181d;
    font:15.5px/1.75 -apple-system,"SF Pro Text","PingFang SC","Microsoft YaHei",sans-serif; }}
  header {{ background:#101827; color:#eef2f7; padding:40px 44px 34px; }}
  header h1 {{ margin:0 0 10px; font-size:25px; font-weight:600; letter-spacing:.2px; }}
  header p {{ margin:4px 0; color:#a9b4c4; font-size:14px; }}
  main {{ max-width:1080px; margin:0 auto; padding:0 24px 72px; }}
  section {{ background:#fff; border:1px solid #e6e8ec; border-radius:10px; padding:28px 32px; margin:20px 0; }}
  h2 {{ font-size:19px; font-weight:600; margin:0 0 16px; }}
  h3 {{ font-size:15.5px; font-weight:600; margin:24px 0 8px; color:#2b3444; }}
  p {{ margin:0 0 14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; margin:14px 0 10px; }}
  th, td {{ border-bottom:1px solid #e9ebef; padding:8px 10px; text-align:right; white-space:nowrap; }}
  th {{ background:#f4f5f7; font-weight:600; color:#3d4655; }}
  td.l, th.l {{ text-align:left; white-space:normal; }}
  tbody tr:hover {{ background:#fafbfc; }}
  .win {{ color:#0b7a5a; }}
  .loss {{ color:#b3261e; }}
  .tie {{ color:#6b7280; }}
  code {{ background:#f1f2f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  pre {{ background:#101827; color:#dfe5ee; padding:15px 17px; border-radius:8px; overflow:auto; font-size:12.5px; line-height:1.6; }}
  figure {{ margin:18px 0 8px; }}
  figcaption {{ color:#6b7280; font-size:12.5px; margin-top:8px; }}
  footer {{ max-width:1080px; margin:0 auto; padding:0 24px 40px; color:#6b7280; font-size:12.5px; }}
  @media (max-width:820px) {{ table {{ font-size:12.5px; }} }}
</style>
</head>
<body>
<header>
  <h1>AgencyBench 评测：标准模式与长程任务模式</h1>
  <p>DeepSeek Harness · deepseek-flash（100 万 token 上下文）· 2026 年 9 月 25 至 26 日</p>
  <p>9 个 scenario / 31 个子任务，两个模式各跑一次；Backend/scenario1 另跑四轮</p>
</header>
<main>

<section class="intro">
  <h2>说明</h2>
  <p>标准模式保留全部历史，只有在接近模型上限时才交给 <code>compaction-basic</code> 压缩。</p>
  <p>长程任务模式把窗口定义在用户提问上。最近 5 个提问保留原文，更早轮次的执行轨迹由记录顶替，
  单个轮次内的轨迹超过 40 万 token 时折回 20 万，折叠边界落在完整的 agent 步上。
  记录不由主 Agent 写：每轮结束，窗口自己发起一次模型调用，把这一轮的轨迹整理成一页结论，
  含任务定义、策略与进展、关键结论、交付与验收四个字段，外加至多几条可复用的经验，
  存进该会话的笔记库 <code>&lt;DSH_HOME&gt;/notes/&lt;sessionId&gt;/</code>，
  连同这次调用问了什么、花了多少 token 一起留档。每条笔记都带它在会话日志里的原文 seq 区间。
  主 Agent 一侧只有读的能力：<code>read_notes</code> 按需取笔记，细节不够时按笔记 id 取全文；
  <code>session_log</code> 按区间取回日志原文，或列出全部记录；<code>fix_note</code> 用来报告写错的笔记。
  提示段只讲怎么读、怎么纠错。</p>
  <p>31 个子任务跑下来，长程任务模式平均 {lh['mean_score']:.2f} 分，标准模式 {std['mean_score']:.2f} 分，3 胜 6 平 0 负。
  成本上长程模式用了 {lh['total_tokens'] / 1e6:.1f} M token，标准模式 {std['total_tokens'] / 1e6:.1f} M；
  平均每次模型调用承载的上下文，长程 {lh['tokens_per_step'] / 1000:.0f} K，标准 {std['tokens_per_step'] / 1000:.0f} K。
  代价落在墙钟上，长程 {lh['total_seconds'] / 60:.0f} 分钟，标准 {std['total_seconds'] / 60:.0f} 分钟。</p>
</section>

<section>
  <h2>1 结果</h2>
  <p>31 个子任务跑完，标准模式平均 {std['mean_score']:.2f} 分，长程任务模式 {lh['mean_score']:.2f} 分，
  9 个 scenario 里长程模式赢 3 个、平 6 个、没有输过。总分差 {score_stats['mean']:+.2f}。</p>
  <table>
    <thead><tr>
      <th class="l">scenario</th><th>子任务</th><th>标准分</th><th>长程分</th><th>分差</th>
      <th>标准 token</th><th>长程 token</th><th>长程/标准</th><th>墙钟 标准 / 长程（分钟）</th>
    </tr></thead>
    <tbody>
{main_rows()}
      <tr><td class="l">合计 / 均值</td><td>31</td><td>{std['mean_score']:.2f}</td><td>{lh['mean_score']:.2f}</td>
        <td class="win">{score_stats['mean']:+.2f}</td><td>{fmt_m(std['total_tokens'])}</td><td>{fmt_m(lh['total_tokens'])}</td>
        <td class="win">{lh['total_tokens'] / std['total_tokens']:.2f}×</td>
        <td>{std['total_seconds'] / 60:.0f} / {lh['total_seconds'] / 60:.0f}</td></tr>
    </tbody>
  </table>
  <figure>
    {score_chart}
    <figcaption>各 scenario 得分。灰条是标准模式，绿条是长程任务模式，满分 10 分。</figcaption>
  </figure>
  <p>三个场景拉开了差距。Backend/scenario3 上长程 9.10 对标准 5.10，这里也是 token 差距最大的地方：标准烧掉 228.9 M，
  长程 52.3 M。标准模式在这个场景派生了 6 个子 agent 会话，整个 run 一共 7 个会话，长程只有 2 个。
  Code/scenario1 上长程 6.00 对标准 4.00，两个模式都没写 benchmark 要求的 <code>docs/subtaskN_rationale.md</code>，
  每个子任务因此丢 5 分，剩下的差别在方程拟合本身，长程做出来 3 个子任务，标准做出来 2 个。
  Backend/scenario1 上长程 10.00 对标准 9.20，差的是一道子任务的半档。</p>
  <p>另外六个场景打平。Code/scenario6 和 scenario7 上两个模式都拿到满分 10 分，研究类的四个场景里，
  Research/scenario2 双方都是 8.00，scenario3 双方 9.67，scenario4 和 scenario5 双方都是 10.00。</p>
  <p>成本上的差别比分数更大。长程模式 161.6 M token，标准模式 365.8 M，前者是后者的 0.44 倍。
  按每次模型调用看，标准模式每步要重读 179.8 K 上下文，长程模式 84.7 K。</p>
  <figure>
    {ratio_chart}
    <figcaption>各 scenario 的 token 比值，长程除以标准。灰条固定在 1.0× 作为基准，绿条低于 1.0 表示长程更省。
    Code/scenario6 是 0.02×，Backend/scenario3 是 0.23×。</figcaption>
  </figure>
</section>

<section>
  <h2>2 成本差在哪里</h2>
  <p>长程模式把若干步之前的完整轨迹换成一条记录块，于是每一步要重读的上下文有了上界。这次运行里长程模式写出 {lh['records']} 条记录，
  触发 {lh['folds']} 次折叠，标准模式两项都是 0。它的上下文峰值始终没到 1M 窗口的压缩阈值，一直在原样重发全部历史。</p>
  <figure>
    {step_chart}
    <figcaption>每次模型调用承载的上下文 token 数。标准模式 {(std['tokens_per_step'] / 1000):.0f} K，长程模式 {(lh['tokens_per_step'] / 1000):.0f} K。</figcaption>
  </figure>
  <p>记录的压缩率很可观。以 Backend/scenario1 那一轮为例，6 条记录分别覆盖 29 K、50 K、108 K、113 K、165 K、86 K token 的轨迹，
  正文只有 6 到 8 千字符。折叠把几十 K 的轨迹压成一条两 K token 左右的块。</p>
  <p>代价有两处。一是折叠块插在历史中段，会打断 prompt 前缀缓存，长程模式的 fresh input 落在 497 K 到 608 K，
  标准模式是 130 K 到 158 K。二是墙钟更长，9 个 scenario 合计 {lh['total_seconds'] / 60:.0f} 分钟对 {std['total_seconds'] / 60:.0f} 分钟，
  其中 Backend/scenario1 与 Code/scenario1 上长程反而更贵，分别是 1.08 倍和 1.66 倍。子任务少、单轮动作小的场景，记录本身的开销摊不薄。</p>
</section>

<section>
  <h2>3 反复跑同一个 scenario</h2>
  <p>Backend/scenario1 上四个 run，每个都是完整协议：5 个子任务加上按 rubric 反馈的重跑，子任务分数取各次尝试的最高分。
  标准模式两轮拿到 36 和 40 分，长程模式拿到 50 和 40 分。有意思的是成本：标准模式两轮的 token 相差 1.95 倍（45.5 M 到 88.6 M），
  长程模式只差 1.13 倍（46.6 M 到 41.4 M）。标准模式的成本随步数和历史长度一起涨，长程模式的上下文有上界，所以贵不起来。</p>
  <table>
    <thead><tr><th class="l">模式</th><th class="l">轮次</th><th>Pass@1</th><th>Pass@2</th>
      <th class="l">各子任务得分</th><th>总 token</th><th>步数</th><th>token/步</th><th>记录/折叠</th><th>墙钟（分钟）</th></tr></thead>
    <tbody>
{round_rows()}
    </tbody>
  </table>
  <figure>
    {rounds_chart}
    <figcaption>四轮的总 token。标准模式从 45.5 M 跳到 88.6 M，长程模式在 46.6 M 和 41.4 M 之间。</figcaption>
  </figure>
  <p>长程模式在第一轮把唯一没满分的子任务补到了 10 分，反馈重跑花了 3.92 M token；标准模式三轮重跑花掉 15.20 M，分数没动。</p>
</section>

<section>
  <h2>4 统计判定</h2>
  <p>分数差 {score_stats['mean']:+.3f}，配对 t 检验 {score_stats['t']:+.2f}（df={score_stats['df']}），
  双尾 p = {score_stats['p']:.3f}，按 α = 0.15 判为显著，85% 置信区间 {score_stats['intervals'][0.85][0]:+.2f} 到 {score_stats['intervals'][0.85][1]:+.2f}。
  精确 Wilcoxon 符号秩检验给出 p = {score_stats['wilcoxon_p']:.3f}。9 个配对里有 {score_stats['ties']} 个差值为 0，
  差异集中在三个场景上。</p>
  <p>token 比的几何平均是 {math.exp(token_stats['mean']):.2f} 倍，也就是典型场景省 {(1 - math.exp(token_stats['mean'])) * 100:.0f}%，
  配对 t 检验 p = {token_stats['p']:.3f}，未达阈值，85% 区间 {token_stats['intervals'][0.85][0]:.2f} 倍到 {token_stats['intervals'][0.85][1]:.2f} 倍。
  比值从 0.02 倍跨到 1.66 倍，方差把均值淹没了。每次调用承载的上下文从 179.8 K 降到 84.7 K，这一项由窗口的构造决定，与抽样无关。</p>
  <h3>分数差</h3>
  <table>
    <tbody>
      <tr><td>均值 ± 标准差</td><td>{score_stats['mean']:+.3f} ± {score_stats['sd']:.3f}</td></tr>
      <tr><td>配对 t / df</td><td>{score_stats['t']:+.2f} / {score_stats['df']}</td></tr>
      <tr><td>双尾 p</td><td>{score_stats['p']:.4f}</td></tr>
      <tr><td>Cohen's d</td><td>{score_stats['cohens_d']:+.2f}</td></tr>
      <tr><td>符号检验</td><td>{score_stats['wins']} 胜 / {score_stats['ties']} 平 / {score_stats['losses']} 负，p = {score_stats['sign_p']:.2f}</td></tr>
      <tr><td>Wilcoxon 符号秩</td><td>W+ = {score_stats['w_plus']:.1f}，n = {score_stats['wilcoxon_n']}，p = {score_stats['wilcoxon_p']:.3f}</td></tr>
    </tbody>
  </table>
  <table>
    <thead><tr><th>置信水平</th><th>均值差区间</th><th>判定</th></tr></thead>
    <tbody>
{interval_rows(score_stats)}
    </tbody>
  </table>
  <h3>token 比</h3>
  <table>
    <tbody>
      <tr><td>几何平均比值</td><td>{math.exp(token_stats['mean']):.3f}×</td></tr>
      <tr><td>对数均值 ± 标准差</td><td>{token_stats['mean']:+.3f} ± {token_stats['sd']:.3f}</td></tr>
      <tr><td>配对 t / df</td><td>{token_stats['t']:+.2f} / {token_stats['df']}</td></tr>
      <tr><td>双尾 p</td><td>{token_stats['p']:.4f}</td></tr>
      <tr><td>符号检验</td><td>{token_stats['losses']} 省 / {token_stats['wins']} 贵 / {token_stats['ties']} 持平，p = {token_stats['sign_p']:.2f}</td></tr>
      <tr><td>Wilcoxon 符号秩</td><td>W+ = {token_stats['w_plus']:.1f}，n = {token_stats['wilcoxon_n']}，p = {token_stats['wilcoxon_p']:.3f}</td></tr>
    </tbody>
  </table>
  <table>
    <thead><tr><th>置信水平</th><th>比值区间</th><th>判定</th></tr></thead>
    <tbody>
{interval_rows(token_stats)}
    </tbody>
  </table>
</section>

<section>
  <h2>5 复现与产物</h2>
  <p>驱动脚本、原始产物与报告本身都在这个仓库里：
  <a href="https://github.com/jacksonwang021219-sudo/agencybench-long-horizon-eval">github.com/jacksonwang021219-sudo/agencybench-long-horizon-eval</a>。
  报告用到的数据文件在 <code>runs/</code> 下，生成脚本在 <code>driver/</code> 下，仓库 README 里写了从零复现的步骤。
  基准本身不入库，按 README 里的命令从 <a href="https://github.com/GAIR-NLP/AgencyBench">GAIR-NLP/AgencyBench</a> 取。</p>
  <pre># 两个模式各跑一遍（9 个 scenario）
python3 driver/run_optionc.py --base-url http://127.0.0.1:&lt;port&gt; --workers 4

# 汇总：分数取各 scenario 的 meta_eval.json，token 取该工作区下所有会话
python3 driver/final_table.py --base-url http://127.0.0.1:&lt;port&gt;

# 单个 scenario
python3 driver/bench_agency.py --scenario ../bench/AgencyBench/AgencyBench-v2/Backend/scenario1 \
  --preset standard --model-name std-be1 --layout subtasks --grader eval-only --second-pass</pre>
  <table>
    <thead><tr><th class="l">产物</th><th class="l">内容</th></tr></thead>
    <tbody>
      <tr><td class="l"><code>runs/optionC/optionC-final.json</code></td><td class="l">9 × 2 的完整结果：分数、token、步数、记录与折叠、墙钟</td></tr>
      <tr><td class="l"><code>runs/optionC/*.json</code>、<code>*.log</code></td><td class="l">每个 run 的逐轮统计和完整日志</td></tr>
      <tr><td class="l"><code>runs/full-*.json</code></td><td class="l">Backend/scenario1 的四轮</td></tr>
      <tr><td class="l"><code>driver/eval_only_shim.py</code></td><td class="l">用官方评分器给外来 agent 的工作区打分</td></tr>
      <tr><td class="l"><code>driver/final_table.py</code>、<code>run_tokens.py</code></td><td class="l">汇总与多会话 token 口径</td></tr>
      <tr><td class="l"><code>report/long-horizon-eval-report.md</code></td><td class="l">长文版本，含逐轮明细与 UltraHorizon 部分</td></tr>
    </tbody>
  </table>
</section>

</main>
<footer>由 <code>driver/make_html_report.py</code> 从 <code>runs/</code> 下的原始产物生成。页面自包含，无外部依赖。
仓库：<a href="https://github.com/jacksonwang021219-sudo/agencybench-long-horizon-eval">agencybench-long-horizon-eval</a></footer>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"[OK] {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
