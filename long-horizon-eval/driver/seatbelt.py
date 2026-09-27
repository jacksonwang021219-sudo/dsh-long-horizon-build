"""Write one macOS seatbelt profile per evaluation arm.

The runner already keeps the grader and the arm's own workspace outside the
benchmark tree, and audits what the agent touched. This adds the enforcement
that audit cannot give: the harness process itself is started under
`sandbox-exec`, with reads and writes of the evaluation workspace denied.

What each profile denies:

* the whole `long-horizon-eval` checkout — the benchmark, the grader source, the
  previous arms' reports, and the notes tooling all live there;
* every OTHER arm's private root, so one arm cannot read another's deliverables.

Everything else stays allowed (`allow default`), which keeps the harness able to
read its own code, the DSH home, the system, and its own workspace. In
`danger-full-access` the bash tool does not call `sandbox-exec` itself, so no
nested profile is ever applied (verified against `packages/shell/bash-sandbox`).

    python3 seatbelt.py --eval-root DIR --own-root DIR --other-root DIR ... --out FILE
"""

from __future__ import annotations

import argparse
from pathlib import Path


def profile(eval_root: str, other_roots: list[str]) -> str:
    """Build one seatbelt profile: allow everything, then deny the evaluation tree."""
    lines = [
        "(version 1)",
        "(allow default)",
        f'(deny file-read* file-write* (subpath "{eval_root}"))',
    ]
    for root in other_roots:
        lines.append(f'(deny file-read* file-write* (subpath "{root}"))')
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--own-root", required=True)
    parser.add_argument("--other-root", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    others = [root for root in args.other_root if root != args.own_root]
    Path(args.out).write_text(profile(str(Path(args.eval_root).resolve()), [str(Path(root).resolve()) for root in others]))
    print(f"{args.out}: deny {1 + len(others)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
