# @deepseek-ai/dsh-context-window

Sliding-window context policy for long-running sessions, carried by the main agent itself.

## What this row adds to a composition

A composition without this row keeps its history the way the harness ships it:
the whole conversation stays on the model-visible surface, `compaction-basic`
compresses it once it passes the model threshold, and `tool-result-pruner`
truncates oversized tool results behind a head and tail. Requests and the traces
that answered them are kept or compressed together; nothing distinguishes one
user request from another.

Mounting this row changes three things.

1. **The window is defined over USER REQUESTS, not over request-and-answer
   pairs.** The two are kept or dropped separately, which is what lets the
   prompts a session was given outlive the work it did in response to them.
2. **A trace is replaced by a record before it leaves.** The record is written
   by one model call the window makes itself at the moment a fold needs it, and
   it stands in the window where the request-and-answer pair it replaced used to
   sit.
3. **The main agent gains a read path and no write path**: `read_notes`,
   `fix_note`, and `session_log`, plus a prompt section that says when to use
   them.

`compaction-basic` and `tool-result-pruner` stay mounted underneath, unchanged.
They are what handles a surface that is still above the model threshold after
the window has folded everything it can.

## The window

The model-visible surface is a window over **user requests**:

- The most recent `retainUserTurns` requests (default 5, counting the request
  being prepared) stay verbatim.
- The execution traces that answered the older ones do not. Each completed
  turn's trace is replaced by the **record for that turn**, and requests older
  than the window leave the surface entirely, replaced by one rebuilt
  `<folded-history>` digest of their records.
- Inside the request in progress, the trace is folded once it passes
  `foldTriggerTokens` (default 400K), back to the most recent `retainTurnTokens`
  (default 200K), at whole agent-step boundaries.
- If the surface is still above the model's own threshold afterwards, the
  inherited `compaction-basic` row compresses it with the harness's existing
  logic.

**Nothing is folded before it is recorded.** A fold can therefore only lose
detail that the session log still holds in full: every record names the log seq
range it accounts for, and the `session_log` tool reads any range back.

## Plugin

Consumes `ctx.tokenMeter`, `ctx.systemPrompt`, and `ctx.tools`. Publishes no
service, so it sits inside the same `isolate` realm as the compaction rows
without adding a realm key.

Register this row **before** `compaction-basic` in the composition: the window
pass then sees the uncompacted surface, and the inherited token-pressure pass
has the final word on the same step.

### What it registers

| Registration | Purpose |
|---|---|
| Prompt section `context:sliding-window` (order 190) | Tells the main agent how to read notes back and how to correct one. Empty for delegated child sessions. |
| `read_notes`, `fix_note`, `session_log` tools | Read the notes, report a wrong note, and read the durable log — raw events by seq range, or the record list. |

### Who writes a note

The window writes it: **one model call** per recording, with no tools and no
agent loop, made at the moment a fold needs the note. The main agent has no
write path at all, so `recordMode` states `recorder` and any other value fails
at load. The request replays the main agent's own prefix (system prompt, tools,
and the model-visible window) and appends the recording contract as the final
user message, so the provider bills it at the cache-hit price and the note
keeper sees exactly what the main agent sees. A reply that cannot be read as a
patch is retried with an actionable hint, up to `recorderRetries`; a recording
that never lands leaves the span unfolded and cites its log range instead.

The contract asks for conclusions in the session's own terms, names only what
the note is about and never who performed it, and forbids a field that merely
points at an earlier note: a note lands in the window where the trace it
replaced used to sit, so it is read as the session's own account of that turn
and has to be usable from the text alone.

The transcript of that call, and what it cost, are the two facts the harness
cannot recover afterwards, so both are written beside the session's notes:

```text
<notesRoot>/<sessionId>/
  index.json                  task list with log ranges and conclusions, topics, types, corrections
  tasks/task-0001.json        one note per recording: the four window fields, open items, reusable lessons
  topics/<slug>.json          topic-level description
  types/<slug>.json           type-level description
  recorder.log                one line per attempt: what was asked, why a reply was rejected
  recorder-usage.jsonl        one line per model call: provider, model, duration, tokens by cache class
```

An index entry carries the note's `keyConclusions`, capped at 400 characters.
That is what makes the next recording's dedup rule applicable and what the
folded-history digest shows for an expired turn; the full text stays in the note.

### Where a record lives

A record is a note file under `<notesRoot>/<sessionId>/tasks/`, written when its
turn is first recorded and read back by id afterwards. The block that stands in
the window for a folded turn is rendered from that note, so the model-facing
text and the stored note cannot drift apart.

Every folded block also carries the note's identity in its own header
(`<turn-summary turn="…" note="…" log-range="…">`) and opens with the same fixed
line, so the main agent reads it as one of its own past answers and can reach the
original events through `session_log`. A fold that could not be recorded cites
its log range alone instead of pretending to carry a note.

The log also carries window-authored blocks for the fold rules that read a
record list rather than a note file. That list is empty in the shipped
composition, because the recorder covers every span it folds, so those rules
never fire; they are kept for a composition whose records arrive some other way.

### When a fold happens

At every pre-step the window plans at most one fold, oldest rule first:

1. **`history-digest`** — turns older than the retained requests, folded into
   one rebuilt digest. Rebuilt from the record list, so previously folded turns
   stay represented.
2. **`turn-trajectory`** — the oldest retained turn's trace, folded into that
   turn's record. Only the part the record covers is folded; an uncovered tail
   stays visible.
3. **`intra-turn`** — the request in progress, once its own trace passes the
   trigger. The fold boundary snaps to a whole agent step, so a tool call never
   separates from its result.

A rule that cannot fold because the agent has not recorded the span returns a
**record obligation** instead: the step then carries a `<window-notice>` naming
the log range to record, and the fold happens on a later step once the record
exists. `recordReminders` (default 2) bounds the asking; an agent that never
records keeps its trace, and the inherited compaction policy owns its pressure.

### When a turn may close

An `agent/turn-stopping` gate requires a main-agent turn that produced work to
have at least one record. It steers a reminder (the same `recordReminders`
bound) instead of closing the turn unrecorded. Delegated child sessions are
exempt.

### Config

| Field | Default | Meaning |
|---|---|---|
| `retainUserTurns` | `5` | User requests kept verbatim, counting the incoming one. |
| `retainTurnTokens` | `200000` | Trajectory tokens a turn keeps after an in-turn fold. |
| `foldTriggerTokens` | `400000` | In-turn trajectory tokens that trigger that fold; must exceed `retainTurnTokens`. |
| `digestMaxTokens` | `8192` | Ceiling on the rebuilt folded-history digest, trimmed from the oldest entry. |
| `recordMaxChars` | `8000` | Ceiling on one recorded summary. |
| `logReadMaxChars` | `60000` | Ceiling on one `session_log` result. |
| `recordReminders` | `2` | Times the window may ask for a record before letting the span or the turn go. |
| `recordMode` | `recorder` | Who writes a turn's record. `recorder` only: the window writes it with one harness call. |
| `policy` | `full` | How much the prompt section says: the recording obligation too (`full`) or the read side only (`minimal`). |
| `recorderMaxTokens` | `2048` | Generation ceiling for one recording call. |
| `recorderView` | `aligned` | Message history one recording call replays: the main agent's window (`aligned`) or the recorded turn alone (`span`). |
| `recorderDedup` | `true` | Whether the recording contract forbids repeating conclusions the notes already carry. |
| `notesRoot` | `<harness home>/notes` | Where the session's notes live, outside every workspace. |
| `recorderRetries` | `5` | Recording attempts before the turn degrades to a log-range citation. |
| `recorderTimeoutMs` | `600000` | Wall-clock ceiling for one recording attempt. |
| `recorderWaitMs` | `120000` | How long the next request waits for an unfinished recording. |
| `recorderCircuitBreaker` | `3` | Consecutive unrecorded turns before the session stops folding and recording for good. |

## Invariants

- **The fold never grows the request.** A replacement block priced at or above
  the span it would replace leaves the surface in place.
- **Unrecorded work is never folded.** Every fold rule requires a record that
  covers the span; a missing record stops the fold at that turn.
- **Step boundaries stay whole.** Folds cut at user-request or agent-step
  boundaries, and the pairing checks in `@deepseek-ai/dsh-compaction` guard both
  edges.
- **The token meter stays exact.** The replacement is preceded by a
  `compaction/prune` shadow-price event naming the replaced range, the shared
  protocol `compaction-tool-result-pruner` also uses. `src/invariant.ts` checks
  that pairing in the durable log.
- **A fold stands for a log range.** Every replacement states the range it
  accounts for, so the log can answer what the surface no longer shows.

## Known Limitations and Deferred Work

- **The dedup rule is an instruction, not a check.** `recorderDedup` puts the
  rule in the contract and the index makes it applicable, but nothing rejects a
  note that repeats a conclusion anyway; repetition is measured outside the
  harness rather than prevented inside it.
- **Note length is soft.** The contract asks for under 2000 characters; a longer
  note is still stored whole. The harness's only defense against a runaway reply
  is `recorderMaxTokens` plus field-level salvage.
- **An expired turn's digest carries truncated conclusions.** The index keeps
  400 characters of each note's `keyConclusions`, so the folded-history block is
  a pointer, not the note: a later turn that needs more reads the note with
  `read_notes` or the events with `session_log`.
- **`reusable_experience` never reaches the window.** It is stored and readable,
  but no replacement block renders it, so it does not influence the next step
  unless something reads the note.
- **The recorder's own cost is not in the session log.** `recorder-usage.jsonl`
  under `notesRoot` is the only accounting; tooling that prices a session must
  read it separately, and a deployment that moves `notesRoot` must move that
  tooling with it.
- **The recorder cannot be routed to another model.** It borrows the session's
  provider and model, which is what makes the prefix cache reusable, so a
  cheaper note-keeping model is not expressible.
- **A note cannot be checked against the terms it was told to keep.** The
  contract asks for the session's own names and for no third-person reference,
  and nothing in the harness enforces either: a note that paraphrases a path or
  names its speaker is stored and rendered like any other.
- **The `agent` fold rules remain without a writer.** The fold planner still
  carries the record-gated path a composition used when the main agent wrote
  records with its own tool call; nothing in the shipped composition produces
  those records, so in practice every span is covered by a note or cited by log
  range. A Code Mode composition therefore folds normally.
