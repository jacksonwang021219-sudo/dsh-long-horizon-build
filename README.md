# DeepSeek Harness — long-horizon sliding-window build

This repository is a complete, runnable checkout of DeepSeek Harness with one
workstream applied on top of it: **the long-horizon task mode**, whose context
policy replaces history with records the harness writes for itself, and the
AgencyBench evaluation of that mode against the standard composition.

Everything needed to run it is here. `node_modules` and build output are not;
`pnpm install` produces them.

## What is in here

| Path | What it is |
|---|---|
| `packages/context/context-window/` | The sliding-window context policy: fold planning, the note recorder, the note store, the read-side tools. |
| `apps/cli/config/agent-presets/long-horizon/` | The shipped `长程任务模式` agent preset, including the `long-horizon-task` skill. |
| `LONG-HORIZON.md` | What the mode changes against upstream, key by key, in Chinese. |
| `long-horizon-eval/report/agencybench-report.html` | The AgencyBench evaluation report (self-contained HTML). |
| `long-horizon-eval/driver/` | The evaluation driver: harness RPC client, run scripts, graders shims, report generator. |
| `packages/`, `apps/`, `docs/`, `vendor/` | The rest of DeepSeek Harness, unmodified. |

## Run it

Requirements: Node `^22.19 || >=24`, pnpm, and a DeepSeek API key.

```sh
pnpm install
export DEEPSEEK_API_KEY=sk-...        # your own key; none is committed here
pnpm dsh --profile web
```

A new session can be started on 长程任务模式 from the preset roster in the Web
UI. For a headless one-shot:

```sh
pnpm dsh --profile headless "your task"
```

The context policy is configured on the preset row, not in code:
`apps/cli/config/agent-presets/long-horizon/agent.cordis.yml`, key
`context-window`. See `packages/context/context-window/README.md` for every key.

## What the long-horizon mode does

The window is defined over **user requests**. The most recent five stay
verbatim; the execution traces that answered older ones do not. They are
replaced by records the window writes for itself, one model call per recording,
made at the moment a fold needs the note.

- A folded record stands where the request-and-answer pair it replaced used to
  sit, so it opens with a fixed line and is read as this session's own account
  of that turn.
- Every record names the log seq range it accounts for, and `session_log` reads
  any range back, so a fold can only lose detail the session log still holds.
- Inside one request, a trace that passes 400K tokens is folded back to the most
  recent 200K, at whole agent-step boundaries.
- The main agent has no write path: it is told how to read a note back and how
  to correct one, never how to write one.

## The evaluation

`long-horizon-eval/report/agencybench-report.html` scores the standard
composition against the long-horizon composition on AgencyBench-v2, per
scenario and per subtask, and accounts for the tokens both spent. It is a
single self-contained file; open it in a browser.

The driver under `long-horizon-eval/driver/` is what produced it. It drives a
running harness over its own HTTP RPC surface and reads each session's durable
log for both scoring and token accounting. It expects the benchmark itself
(AgencyBench-v2) to be present at `long-horizon-eval/bench/`, which is not
redistributed here.

## Provenance

The harness is DeepSeek Harness. The long-horizon work above it, and the
evaluation, are not upstream. Nothing in this repository is a claim about
upstream behaviour.

## License

DeepSeek Harness is distributed under the license in `LICENSE`. The
long-horizon additions and the evaluation code follow the same terms.
