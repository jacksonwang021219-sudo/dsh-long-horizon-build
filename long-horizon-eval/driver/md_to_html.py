"""Render one of the evaluation reports as a standalone styled HTML page.

Dependency-free Markdown subset: headings, tables (pipes inside inline code are
honoured), fenced code, blockquotes, bullet and numbered lists, inline code,
bold, and links. The output is one self-contained file with inline CSS — no
scripts, no CDN — so it opens from disk in any browser.

    python3 md_to_html.py ../report/v1-v2-aligned-recorder-report.md \
        --title "V1 / V2 记录器对比评测报告" --subtitle "..." --out ../report/v1-v2-aligned-recorder-report.html
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

CSS = """
:root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#0f766e; --warn:#b45309; }
* { box-sizing:border-box; }
body { margin:0; padding:0 0 64px; font:15px/1.7 -apple-system,"SF Pro Text","PingFang SC","Microsoft YaHei",sans-serif;
       color:var(--ink); background:#f9fafb; }
header { background:#0f172a; color:#fff; padding:36px 40px 28px; }
header h1 { margin:0 0 8px; font-size:26px; letter-spacing:.2px; }
header p { margin:4px 0; color:#cbd5e1; font-size:14px; }
main { max-width:1100px; margin:0 auto; padding:0 24px; }
h2 { font-size:21px; margin:34px 0 14px; padding-bottom:10px; border-bottom:2px solid var(--accent); display:inline-block; }
h3 { font-size:16.5px; margin:24px 0 10px; color:#0f172a; }
h4 { font-size:15px; margin:18px 0 8px; color:#374151; }
p, li { color:#1f2937; }
blockquote { background:#ecfdf5; border-left:4px solid var(--accent); padding:12px 18px; border-radius:0 8px 8px 0;
             margin:16px 0; color:#134e4a; }
blockquote p { margin:4px 0; color:#134e4a; }
hr { border:0; border-top:1px solid var(--line); margin:28px 0; }
table { width:100%; border-collapse:collapse; font-size:13.5px; margin:12px 0 8px; background:#fff; }
th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { background:#f3f4f6; font-weight:600; color:#374151; }
tbody tr:hover { background:#f9fafb; }
code { background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12.8px;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
pre { background:#0f172a; color:#e2e8f0; padding:14px 16px; border-radius:8px; overflow:auto; font-size:12.8px; }
pre code { background:none; color:inherit; padding:0; }
a { color:var(--accent); }
footer { max-width:1100px; margin:32px auto 0; padding:0 24px; color:var(--muted); font-size:12.5px; }
strong { color:#0f172a; }
"""


def split_row(line: str) -> list[str]:
    """Split one table row on unescaped pipes that are outside inline code."""
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == '\\' and index + 1 < len(line) and line[index + 1] == '|':
            current.append('|')
            index += 2
            continue
        if char == '`':
            in_code = not in_code
            current.append(char)
            index += 1
            continue
        if char == '|' and not in_code:
            cells.append(''.join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append(''.join(current))
    if cells and cells[0].strip() == '':
        cells = cells[1:]
    if cells and cells[-1].strip() == '':
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def inline(text: str) -> str:
    """Render inline Markdown: code spans first, then bold, links, escapes."""
    placeholders: list[str] = []

    def hold(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    def code_span(match: re.Match[str]) -> str:
        return hold(f"<code>{html.escape(match.group(1))}</code>")

    text = re.sub(r"`([^`]+)`", code_span, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def convert(markdown: str) -> str:
    """Convert the report's Markdown subset to HTML."""
    out: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith('```'):
            block: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith('```'):
                block.append(lines[index])
                index += 1
            index += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            continue
        if stripped.startswith('# '):
            index += 1
            continue
        if stripped.startswith('### '):
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
            index += 1
            continue
        if stripped.startswith('## '):
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
            index += 1
            continue
        if stripped.startswith('> '):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith('>'):
                quote.append(lines[index].strip().lstrip('>').strip())
                index += 1
            body = "\n".join(quote)
            out.append(f"<blockquote>{paragraphs(body)}</blockquote>")
            continue
        if stripped.startswith('|') and index + 1 < len(lines) and set(lines[index + 1].strip()) <= set('|-: '):
            header = split_row(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append(split_row(lines[index].strip()))
                index += 1
            head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if re.match(r"^[-*] ", stripped):
            items: list[str] = []
            while index < len(lines) and re.match(r"^[-*] ", lines[index].strip()):
                items.append(f"<li>{inline(lines[index].strip()[2:])}</li>")
                index += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+\. ", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\. ", lines[index].strip()):
                items.append(f"<li>{inline(re.sub(r'^[0-9]+. ', '', lines[index].strip()))}</li>")
                index += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        if stripped == '':
            index += 1
            continue
        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^(#|>|\||```|[-*] |\d+\. )", lines[index].strip()
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(paragraphs(" ".join(paragraph)))
    return "\n".join(out)


def paragraphs(text: str) -> str:
    """Wrap blank-line-separated text in paragraph tags."""
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return "".join(f"<p>{inline(part)}</p>" for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--title", default="")
    parser.add_argument("--subtitle", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    body = convert(source.read_text())
    title = args.title or source.stem
    subs = "".join(f"<p>{inline(line)}</p>" for line in args.subtitle)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  {subs}
</header>
<main>
{body}
</main>
<footer>由 {html.escape(source.name)} 生成 · {len(body.splitlines())} 行正文</footer>
</body>
</html>
"""
    Path(args.out).write_text(page)
    print(f"{args.out} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
