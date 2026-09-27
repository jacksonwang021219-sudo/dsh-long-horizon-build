# Agent Note: Turn records written by the agent or by the window's recorder

Status: implemented

## Problem

The long-horizon context policy folded everything older than the newest user
turns into one block, and rebuilt that block by spawning a **fresh subagent per
user turn** (`dsh-context-window/assembler`, `ASSEMBLY_SYSTEM_LOGIC`). Three
properties of that design were wrong for a window defined over requests:

1. The window kept whole request-and-answer turns. Traces of older turns stayed
   on the model surface long after they had been superseded, so the retained
   window grew without bound in the dimension that matters (agent steps).
2. Nothing durable recorded what a folded turn had established. The replacement
   was a re-derivation, so the same facts were re-discovered (and re-paid for)
   every turn, and an assembly failure silently kept the full surface.
3. The fold had no token rule inside one long turn, so one runaway turn could
   grow to the model's ceiling before anything happened.

## Decision

The window keeps the newest N **user requests**; the traces that answered older
requests are replaced by records, and `recordMode` chooses who writes one:

- **`agent`** (default): the main agent records itself. A prompt section
  (`context:sliding-window`) plus two tools (`record_turn`, `session_log`) carry
  what the assembly agent used to know; `agent/turn-stopping` gates the turn, and
  the gate steers a reminder instead of forcing a record.
- **`recorder`**: the window records, in **one model call** with no tool and no
  agent loop, at the moment a fold needs the record. The main agent keeps only
  the read side (`read_notes`, `fix_note`, `session_log`) and is never asked to
  write, so a turn's context bound does not depend on the model cooperating.

Whichever mode writes it, the rest of the policy is the same:

- A record names the log seq range it accounts for, which is what makes the log
  indexable by record. In `agent` mode it is a top-level `record_turn` tool
  result whose structured payload rides the durable `tool/result.meta`; in
  `recorder` mode it is a note page under `notesRoot`, outside every workspace.
- The session's `index.json` carries each note's `keyConclusions` (capped at 400
  characters), so the folded-history digest of an expired turn states what that
  turn concluded rather than only its title, and the next recording can enforce
  its dedup rule against what is already recorded.
- **A span is never folded before it is recorded.** Every rule requires a record
  covering the span; when one is missing the window either records it
  synchronously or replaces it with a block citing the bare log range.
- An in-turn rule folds a turn's own trace once it passes a trigger, back to a
  smaller retained window, at whole agent-step boundaries.

### The recorder's request

`recorderView` decides what the recording call replays:

- **`aligned`** (default): the main agent's own `system` + `tools` + the
  model-visible window. Only the tail differs from the conversation's own
  request, so the provider serves the shared prefix from its cache (measured:
  ~90% of the prompt) and the note keeper judges this turn against exactly what
  the main agent can still see.
- **`span`**: the recorded turn's messages alone. The request diverges at its
  first message, so only the system prompt and tool block are cacheable
  (measured: ~65% of the prompt is billed at the miss price), and the recorder
  cannot see which conclusions an earlier summary already carries —
  `recorderDedup` is therefore unenforceable in this view.

## Alternatives considered

### Why not a dedicated `context-window/record` session event?

A dedicated event type is the obvious durable channel for a record, and it was
the first design. It loses on the persistence catalog: a new `SessionEventMap`
member is required-on-read by default, so every build that does not know the type
refuses a log containing it, and the fork's records would make sessions
unreadable by a stock build. `tool/result.meta` is already durable, JSON-validated,
tool-owned, and invisible to the core's reconstruction rules, so records survive
both the fold and a stock build reading the same session. The cost is a
documented limitation: `presentationMeta` is projected only for top-level calls,
so a composition running tools through Code Mode leaves no records — and then the
window simply refuses to fold.

### Why not keep the assembly agent and add its log-range field?

Reusing the existing per-turn assembly agent and only adding a log-range field
was cheaper to build, and it was rejected because it keeps paying for the same
knowledge twice: a second model call per user turn, re-derived from the workspace
every time, with no durable memory of what any previous turn established. The
agent also had no way to answer "what did you already fold?" — its replacement
block was prose, not an index — which is exactly the gap the records close.

### Why `recorder` mode does not fork a child session

A forked child session would carry its own log, quota, and title, and its tokens
would land in a different session's accounting than the work they describe. The
recorder instead borrows the parent session's route and id for one call with
`purpose: 'context-record'`, so it shares the parent's warmed prefix cache while
staying out of the parent's log — which is why its cost is written to
`recorder-usage.jsonl` beside the notes instead of being derivable from the
session log.

## Consequences

- Token pressure inside a turn is bounded by `foldTriggerTokens`; the retained
  tail after a fold is `retainTurnTokens`.
- In `agent` mode each user turn costs one extra model step (the record call,
  and the gate's reminder when the agent records early). In `recorder` mode it
  costs one extra model call, whose price depends entirely on `recorderView`:
  the aligned view is billed mostly at the cache-hit price, the span view mostly
  at the miss price. Either way the cost is only repaid once the window actually
  folds, i.e. from the sixth user request onward under the shipped defaults.
- `recorder.log` and `recorder-usage.jsonl` are the recorder's only durable
  traces: what it was asked, why a reply was rejected, and what each call cost.
- The fold's replacement is built from records, so a folded span can always be
  read back by log range (`session_log`), including events no longer on the
  surface.
- Unrecorded work is never folded: a recording that keeps failing trips the
  circuit breaker after `recorderCircuitBreaker` turns, freezes folding for the
  session with a visible notice, and the inherited `compaction-basic` policy
  owns the pressure from then on.

## Verification

- `packages/context/context-window/tests/window.spec.ts` covers every rule:
  digest, trajectory, intra-turn record-first-then-fold, the refusal to fold an
  unrecorded turn, the index's truncated conclusions, the recorder's usage log,
  and the contract's dedup rule being present only when `recorderDedup` is on.
- `src/invariant.ts` checks the fold pairing in the durable log: a window fold
  block must be immediately preceded by the `compaction/prune` that prices it
  and must replace exactly the range it priced.
- Live harness runs against the stress presets (retain 1 request, 3000/6000
  token thresholds): four turns produced four notes and three folds under both
  `recorderView` values, with the digest carrying each note's conclusions. The
  same runs measured the cache split the views predict — aligned 12.5K of 13.6K
  prompt tokens cached per call, span 9.0K of 13.8K.
