/**
 * Sliding-window selection over the live session surface.
 *
 * A window here is defined over USER REQUESTS: the newest requests stay on the
 * model-visible surface, while the execution traces that answered the older
 * ones are replaced by the records the agent wrote for them. Nothing is ever
 * folded out of the surface before the agent recorded it, so a fold can only
 * lose detail that the log still holds in full.
 *
 * @module @deepseek-ai/dsh-context-window/window
 */

import { toolPairingBalancedAfter, toolPairingBalancedBefore } from '@deepseek-ai/dsh-compaction'
import type { ContentBlock, UserMessage } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { ResolvedContextWindowConfig } from './config.ts'
import {
  coverageEnd,
  isRecordEvent,
  nextRecordStart,
  recordOfEvent,
  recordsForTurn,
  renderHistoryDigest,
  renderRecordBlock,
  renderUnrecordedBlock,
} from './records.ts'
import type { ContextRecord, DigestEntry } from './records.ts'

/** The message-source plugin name every window-authored block records. */
export const WINDOW_SOURCE_PLUGIN = 'dsh-context-window'

/** One inclusive positional span of the current surface to fold away. */
export interface WindowShadowSpan {
  /** Inclusive first surface-node seq of the span. */
  readonly start: number
  /** Inclusive last surface-node seq of the span. */
  readonly end: number
  /** Every folded seq, in surface order — the authoritative shadowed set. */
  readonly shadowedSeqs: readonly number[]
  /** Heuristic price of the span under the token-meter's fixed estimator. */
  readonly shadowedTokens: number
}

/** Why one fold is happening. */
export type FoldReason = 'history-digest' | 'turn-trajectory' | 'intra-turn'

/** One fold the window wants to perform on the current step. */
export interface FoldPlan {
  /** Which rule selected the span. */
  readonly reason: FoldReason
  /** Replacement content, framed for the model. */
  readonly text: string
  /** The span to replace with one block. */
  readonly span: WindowShadowSpan
  /**
   * The single turn whose trace this fold replaces, when the fold covers one
   * turn. `undefined` for the multi-turn digest, which is rebuilt from records
   * rather than recorded again.
   */
  readonly turn?: number | undefined
}

/** A record the agent must write before the window may fold. */
export interface RecordObligation {
  /** The turn the record belongs to. */
  readonly turn: number
  /** Inclusive first log seq the record must account for. */
  readonly start: number
  /** Inclusive last log seq the record must account for. */
  readonly end: number
  /** Why the obligation exists, for the injected instruction. */
  readonly reason: 'intra-turn'
}

/** What the window pass decided to do on this step. */
export type WindowDecision =
  | { readonly kind: 'fold'; readonly plan: FoldPlan }
  | { readonly kind: 'record-first'; readonly obligation: RecordObligation; readonly turnTokens: number }
  | { readonly kind: 'none' }

/**
 * Join the text blocks of one content array, dropping non-text blocks.
 * @param blocks - the content blocks to read.
 * @returns the joined text, trimmed, or an empty string without any text block.
 */
export function contentText(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type !== 'text') continue
    if (block.text.length > 0) parts.push(block.text)
  }
  return parts.join('\n').trim()
}

/**
 * The prompt text of the request being prepared, or `undefined` when this step
 * carries no human-authored message — an injected context append, for example.
 * @param messages - the messages claimed for the proposed step.
 * @returns the joined prompt text, or `undefined` without one.
 */
export function humanPromptText(messages: readonly UserMessage[]): string | undefined {
  const parts: string[] = []
  for (const message of messages) {
    if (message.source.kind !== 'user') continue
    const text = contentText(message.content)
    if (text.length > 0) parts.push(text)
  }
  return parts.length === 0 ? undefined : parts.join('\n\n')
}

/**
 * Whether one surface event is a human-authored prompt rather than injected content.
 * @param event - the event to test, or `undefined` for a seq with no event.
 * @returns true for a user-authored prompt.
 */
export function isHumanPrompt(event: SessionEvent | undefined): boolean {
  if (event === undefined || event.type !== 'user/message') return false
  return event.data.source.kind === 'user'
}

/**
 * Whether one surface event is a block this plugin wrote.
 * @param event - the event to test, or `undefined` for a seq with no event.
 * @returns true for a window block.
 */
export function isWindowBlock(event: SessionEvent | undefined): boolean {
  if (event === undefined || event.type !== 'user/message') return false
  const source = event.data.source
  return source.kind === 'plugin' && source.plugin === WINDOW_SOURCE_PLUGIN
}

/** What one surface node is, for window accounting. */
type SurfaceRole = 'prompt' | 'window-block' | 'record' | 'trace'

/** One classified surface node. */
interface SurfaceNode {
  readonly seq: number
  readonly index: number
  readonly role: SurfaceRole
  /** Owning turn; `0` for nodes appended before the first turn opened. */
  readonly turn: number
  readonly tokens: number
}

/** Turn and step boundaries of one session log, scanned incrementally. */
interface TurnIndex {
  /** Number of log events already scanned. */
  scanned: number
  /** Opening seq of each turn, in turn order. */
  readonly turnStarts: { seq: number; turn: number }[]
  /** Opening seq of each step, in step order. */
  readonly stepStarts: { seq: number; turn: number; step: number }[]
}

const turnIndexes = new WeakMap<Session, TurnIndex>()

/** Read or extend the incremental turn/step boundary index of one session. */
function turnIndex(session: Session): TurnIndex {
  const events = session.events
  let index = turnIndexes.get(session)
  if (index === undefined || index.scanned > events.length) {
    index = { scanned: 0, turnStarts: [], stepStarts: [] }
    turnIndexes.set(session, index)
  }
  for (let position = index.scanned; position < events.length; position += 1) {
    const event = events[position]
    if (event === undefined) continue
    if (event.type === 'turn/start') index.turnStarts.push({ seq: event.seq, turn: event.data.turn })
    if (event.type === 'step/start') {
      index.stepStarts.push({ seq: event.seq, turn: event.data.turn, step: event.data.step })
    }
  }
  index.scanned = events.length
  return index
}

/** The turn owning one log seq: the newest turn opened before it, or `0`. */
function turnOfSeq(index: TurnIndex, seq: number): number {
  let turn = 0
  for (const start of index.turnStarts) {
    if (start.seq > seq) break
    turn = start.turn
  }
  return turn
}

/** The newest turn the log has opened, or `0` before the first turn. */
function currentTurn(index: TurnIndex): number {
  const last = index.turnStarts[index.turnStarts.length - 1]
  return last === undefined ? 0 : last.turn
}

/** Classify every node of the current surface. */
function surfaceNodes(session: Session, priced: ReadonlyMap<number, number>): SurfaceNode[] {
  const index = turnIndex(session)
  const nodes: SurfaceNode[] = []
  const seqs = session.surface.nodes
  for (let position = 0; position < seqs.length; position += 1) {
    const seq = seqs[position]
    if (seq === undefined) continue
    const event = session.events[seq]
    const record = recordOfEvent(event)
    const role: SurfaceRole = isHumanPrompt(event)
      ? 'prompt'
      : isWindowBlock(event)
        ? 'window-block'
        : record === undefined ? 'trace' : 'record'
    nodes.push({
      seq,
      index: position,
      role,
      turn: record === undefined ? turnOfSeq(index, seq) : record.turn,
      tokens: priced.get(seq) ?? 0,
    })
  }
  return nodes
}

/** Sum the priced tokens of a node slice. */
function sumTokens(nodes: readonly SurfaceNode[]): number {
  let total = 0
  for (const node of nodes) total += node.tokens
  return total
}

/** Build the span descriptor for an inclusive run of surface nodes. */
function spanOf(nodes: readonly SurfaceNode[], from: number, to: number): WindowShadowSpan | null {
  if (from > to || from < 0 || to >= nodes.length) return null
  const slice = nodes.slice(from, to + 1)
  const first = slice[0]
  const last = slice[slice.length - 1]
  if (first === undefined || last === undefined) return null
  return {
    start: first.seq,
    end: last.seq,
    shadowedSeqs: slice.map(node => node.seq),
    shadowedTokens: sumTokens(slice),
  }
}

/** Whether both edges of a span are safe cuts for a single replacement node. */
function balancedEdges(session: Session, span: WindowShadowSpan): boolean {
  return toolPairingBalancedBefore(session, span.start)
    && toolPairingBalancedAfter(session, span.end)
}

/** Everything one window pass needs to decide. */
export interface WindowPlanInput {
  /** Validated row policy. */
  readonly config: ResolvedContextWindowConfig
  /** Every record the session log holds. */
  readonly records: readonly ContextRecord[]
  /** Current surface price per node seq, from the token meter. */
  readonly priced: ReadonlyMap<number, number>
  /** Estimated tokens of an arbitrary text under the same estimator. */
  readonly estimate: (text: string) => number
  /** Whether the step being prepared will append a new human request. */
  readonly incomingPrompt: boolean
  /**
   * Who writes a record. In `recorder` mode the window records at fold time, so
   * a span is foldable before any record exists for it; `agent` keeps the
   * original rule that a span is folded only once a record covers it.
   */
  readonly recordMode: 'agent' | 'recorder'
  /**
   * The in-turn record obligation the window already asked for, when one is
   * outstanding. A fold reuses its boundary instead of recomputing one, so the
   * boundary cannot walk forward while the agent is writing the record.
   */
  readonly pending?: { readonly turn: number; readonly end: number } | undefined
}

/**
 * Decide what the window does on the step being prepared: fold a span, demand
 * the record a fold depends on, or nothing.
 *
 * Preference order is deliberately oldest-first — the folded-history digest,
 * then the oldest retained turn's trace, then the turn in progress — so one
 * pass converges towards the requested window without ever folding a span the
 * agent has not recorded.
 * @param session - session whose surface and log to examine.
 * @param input - policy, records, prices, and whether a request is arriving.
 * @returns this step's window decision.
 */
export function planWindow(session: Session, input: WindowPlanInput): WindowDecision {
  const nodes = surfaceNodes(session, input.priced)
  if (nodes.length === 0) return { kind: 'none' }
  const digest = planHistoryDigest(session, nodes, input)
  if (digest !== null) return { kind: 'fold', plan: digest }
  const trajectory = planTurnTrajectory(session, nodes, input)
  if (trajectory !== null) return { kind: 'fold', plan: trajectory }
  const inTurn = planIntraTurn(session, nodes, input)
  return inTurn ?? { kind: 'none' }
}

/**
 * Fold every turn older than the retained requests — requests, traces and
 * older blocks together — into one rebuilt digest.
 */
function planHistoryDigest(
  session: Session,
  nodes: readonly SurfaceNode[],
  input: WindowPlanInput,
): FoldPlan | null {
  const prompts = nodes.filter(node => node.role === 'prompt')
  const keep = Math.max(1, input.config.retainUserTurns - (input.incomingPrompt ? 1 : 0))
  if (prompts.length <= keep) return null
  const firstRetained = prompts.length - keep

  // Only turns the agent recorded may leave the surface. A turn with an
  // unrecorded trace stops the fold there, so its trace survives the pass.
  let stop = 0
  for (let position = 0; position < firstRetained; position += 1) {
    const prompt = prompts[position]
    const next = prompts[position + 1]
    if (prompt === undefined || next === undefined) break
    const traced = nodes.slice(prompt.index + 1, next.index).some(node => node.role === 'trace')
    const recorded = input.recordMode === 'recorder'
      || coverageEnd(input.records, prompt.turn) !== undefined
    if (traced && !recorded) break
    stop = position + 1
  }
  if (stop === 0) return null

  const head = nodes.findIndex(node => node.role === 'prompt' || node.role === 'window-block')
  const lastRemoved = prompts[stop - 1]
  const firstKept = prompts[stop]
  if (head < 0 || lastRemoved === undefined || firstKept === undefined) return null
  const span = spanOf(nodes, head, firstKept.index - 1)
  if (span === null || !balancedEdges(session, span)) return null

  const oldestRetainedTurn = firstKept.turn
  const entries = digestEntries(nodes, input.records, oldestRetainedTurn)
  if (entries.length === 0) return null
  const text = renderHistoryDigest(entries, input.config.digestMaxTokens, input.estimate)
  if (text.length === 0) return null
  return { reason: 'history-digest', text, span }
}

/**
 * Describe every turn older than the oldest retained one, oldest first, from
 * the records it wrote and the requests still on the surface.
 */
function digestEntries(
  nodes: readonly SurfaceNode[],
  records: readonly ContextRecord[],
  oldestRetainedTurn: number,
): DigestEntry[] {
  const prompts = nodes.filter(node => node.role === 'prompt')
  const ranges = new Map<number, { start: number; end: number; summary?: string }>()
  for (const prompt of prompts) {
    if (prompt.turn >= oldestRetainedTurn) continue
    const following = prompts.find(candidate => candidate.turn > prompt.turn)
    ranges.set(prompt.turn, {
      start: prompt.seq,
      end: following === undefined ? prompt.seq : following.seq - 1,
    })
  }
  for (const record of records) {
    if (record.turn >= oldestRetainedTurn) continue
    const existing = ranges.get(record.turn)
    if (existing === undefined) {
      ranges.set(record.turn, { start: record.start, end: record.end, summary: record.summary })
      continue
    }
    existing.start = Math.min(existing.start, record.start)
    existing.end = Math.max(existing.end, record.end)
    existing.summary = record.summary
  }
  const turns = [...ranges.keys()].sort((left, right) => left - right)
  const entries: DigestEntry[] = []
  for (const turn of turns) {
    const range = ranges.get(turn)
    if (range === undefined) continue
    entries.push({
      turn,
      start: range.start,
      end: range.end,
      ...range.summary === undefined ? {} : { summary: range.summary },
    })
  }
  return entries
}

/**
 * Replace one completed turn's execution trace with the records written for
 * it, leaving whatever the records do not cover in place.
 */
function planTurnTrajectory(
  session: Session,
  nodes: readonly SurfaceNode[],
  input: WindowPlanInput,
): FoldPlan | null {
  const prompts = nodes.filter(node => node.role === 'prompt')
  const live = input.incomingPrompt ? prompts.length - 1 : prompts.length - 2
  for (let position = 0; position <= live; position += 1) {
    const prompt = prompts[position]
    const next = prompts[position + 1]
    if (prompt === undefined || next === undefined) continue
    const records = recordsForTurn(input.records, prompt.turn)
    // The recorder writes the record as it folds, so it needs none in hand.
    const recorded = input.recordMode === 'recorder'
    if (!recorded && records.length === 0) continue
    const covered = coverageEnd(input.records, prompt.turn)
    if (!recorded && covered === undefined) continue
    const from = prompt.index + 1
    let end = -1
    for (let cursor = from; cursor < next.index; cursor += 1) {
      const node = nodes[cursor]
      if (node === undefined) break
      if (covered !== undefined && node.seq > covered) break
      if (node.role === 'record') continue
      if (recordOfSeqs(input.records, node.seq)) continue
      end = cursor
    }
    if (end < from) continue
    let cut = end
    while (cut >= from
      // oxlint-disable-next-line typescript/no-non-null-assertion -- cut indexes the checked slice.
      && !toolPairingBalancedAfter(session, nodes[cut]!.seq)) cut -= 1
    if (cut < from) continue
    const folded = nodes.slice(from, cut + 1)
    if (!folded.some(node => node.role === 'trace')) continue
    const span = spanOf(nodes, from, cut)
    if (span === null || !balancedEdges(session, span)) continue
    const text = recorded
      ? renderRecorderPlaceholder(prompt.turn, span)
      : renderRecordBlock(prompt.turn, records)
    // In recorder mode the replacement is replaced again once it is written, so
    // only a written record is priced against the span.
    if (!recorded && input.estimate(text) >= span.shadowedTokens) continue
    return { reason: 'turn-trajectory', text, span, turn: prompt.turn }
  }
  return null
}

/** Whether one seq carries a record tool result (its own trace node). */
function recordOfSeqs(records: readonly ContextRecord[], seq: number): boolean {
  return records.some(record => record.seq === seq)
}

/**
 * Fold the head of the turn in progress once its own trace exceeds the
 * trigger, back to `retainTurnTokens`. Returns a record obligation instead of
 * a fold while the agent has not recorded the range yet.
 */
function planIntraTurn(
  session: Session,
  nodes: readonly SurfaceNode[],
  input: WindowPlanInput,
): WindowDecision | null {
  const index = turnIndex(session)
  const turn = currentTurn(index)
  if (turn === 0) return null
  const turnStart = index.turnStarts[index.turnStarts.length - 1]
  if (turnStart === undefined) return null
  const owned = nodes.filter(node => node.seq > turnStart.seq)
  const turnTokens = sumTokens(owned)
  if (turnTokens < input.config.foldTriggerTokens) return null

  const prompts = owned.filter(node => node.role === 'prompt')
  const prompt = prompts[prompts.length - 1]
  if (prompt === undefined) return null
  const head = prompt.index + 1
  if (head >= nodes.length) return null

  // Fold whole agent steps: a cut is the last surface node of a step, so a
  // tool call never separates from its result.
  const cuts: number[] = []
  const steps = index.stepStarts.filter(start => start.turn === turn)
  for (let position = 0; position < steps.length - 1; position += 1) {
    const nextStep = steps[position + 1]
    if (nextStep === undefined) continue
    let candidate = -1
    for (let cursor = head; cursor < nodes.length; cursor += 1) {
      const node = nodes[cursor]
      if (node === undefined || node.seq >= nextStep.seq) break
      candidate = cursor
    }
    if (candidate >= head) cuts.push(candidate)
  }
  if (cuts.length === 0) return null

  // The first balanced cut that brings the retained tail under the target keeps
  // the most history; when a single step is already bigger than the target, no
  // cut can, and folding up to the last one is the closest rounding to whole
  // steps the policy can reach without cutting a step in half.
  let chosen = -1
  let fallback = -1
  for (const cut of cuts) {
    const node = nodes[cut]
    if (node === undefined) continue
    if (!toolPairingBalancedAfter(session, node.seq)) continue
    fallback = cut
    if (sumTokens(nodes.slice(cut + 1)) > input.config.retainTurnTokens) continue
    chosen = cut
    break
  }
  if (chosen < 0) chosen = fallback
  // An outstanding obligation pins the boundary the agent is recording against.
  const pinned = input.pending === undefined || input.pending.turn !== turn
    ? -1
    : nodes.findIndex(node => node.seq === input.pending?.end)
  if (pinned >= head) chosen = pinned
  if (chosen < 0) return null

  const cutNode = nodes[chosen]
  if (cutNode === undefined) return null
  const covered = coverageEnd(input.records, turn)
  if (covered === undefined || covered < cutNode.seq) {
    return {
      kind: 'record-first',
      turnTokens,
      obligation: {
        turn,
        start: nextRecordStart(input.records, turn, prompt.seq),
        end: cutNode.seq,
        reason: 'intra-turn',
      },
    }
  }
  const span = spanOf(nodes, head, chosen)
  if (span === null || !balancedEdges(session, span)) return null
  const text = renderRecordBlock(turn, recordsForTurn(input.records, turn))
  if (input.estimate(text) >= span.shadowedTokens) return null
  return { kind: 'fold', plan: { reason: 'intra-turn', text, span, turn } }
}

/**
 * Plan the fold the window performs when the agent never wrote the record its
 * in-turn obligation asked for.
 *
 * The obligation's boundary is reused exactly, so the fold the agent was told
 * about is the fold that happens; only the replacement differs, and it names
 * the uncovered log range instead of a summary.
 * @param session - session whose surface to fold.
 * @param obligation - the boundary the window asked the agent to record.
 * @param priced - current surface price per node seq.
 * @param estimate - fixed estimator used for the never-grow guard.
 * @returns the fold, or `null` when the boundary is no longer foldable.
 */
export function foldPlanForObligation(
  session: Session,
  obligation: RecordObligation,
  priced: ReadonlyMap<number, number>,
  estimate: (text: string) => number,
): FoldPlan | null {
  const nodes = surfaceNodes(session, priced)
  const prompts = nodes.filter(node => node.role === 'prompt')
  const prompt = prompts.findLast(node => node.turn === obligation.turn)
  if (prompt === undefined) return null
  const head = prompt.index + 1
  let cut = nodes.findIndex(node => node.seq === obligation.end)
  if (cut < head) return null
  while (cut >= head) {
    const node = nodes[cut]
    if (node !== undefined && toolPairingBalancedAfter(session, node.seq)) break
    cut -= 1
  }
  if (cut < head) return null
  const span = spanOf(nodes, head, cut)
  if (span === null || !balancedEdges(session, span)) return null
  const text = renderUnrecordedBlock(obligation.turn, obligation.start, obligation.end, span.shadowedTokens)
  if (estimate(text) >= span.shadowedTokens) return null
  return { reason: 'intra-turn', text, span, turn: obligation.turn }
}

/** Whether one event is a record tool result, re-exported for callers that classify surfaces. */
export { isRecordEvent }

/**
 * The placeholder a recorder-mode fold carries until the recorder has run: the
 * planner is synchronous, so the body is filled in by the caller before the
 * append.
 * @param turn - the turn being recorded.
 * @param span - the span being replaced.
 * @returns the framed placeholder text.
 */
export function renderRecorderPlaceholder(turn: number, span: WindowShadowSpan): string {
  return `<turn-record turn="${String(turn)}" log-range="${String(span.start)}-${String(span.end)}" tokens="${String(span.shadowedTokens)}" kind="turn" source="recorder">pending</turn-record>`
}

/**
 * Frame one recorder-written body as the block that replaces a span.
 * @param turn - the turn the record belongs to.
 * @param span - the span being replaced.
 * @param body - the recorder's text.
 * @returns the framed replacement text.
 */
export function renderRecorderBlock(
  turn: number,
  span: WindowShadowSpan,
  body: string,
): string {
  return [
    `<turn-record turn="${String(turn)}" log-range="${String(span.start)}-${String(span.end)}" tokens="${String(span.shadowedTokens)}" kind="turn" source="recorder">`,
    `This turn's execution trace left the visible window and was recorded for you. The original events stay in the session log (seq ${String(span.start)}-${String(span.end)}).`,
    '',
    body,
    '</turn-record>',
  ].join('\n')
}

/**
 * The newest turn the session log has opened, or `0` before the first turn.
 * @param session - session whose log to read.
 * @returns the current turn number.
 */
export function currentTurnOf(session: Session): number {
  return currentTurn(turnIndex(session))
}

/**
 * The seq of one turn's opening `turn/start` event.
 * @param session - session whose log to read.
 * @param turn - the turn to locate.
 * @returns the opening seq, or `undefined` when the log never opened that turn.
 */
export function turnStartSeqOf(session: Session, turn: number): number | undefined {
  const starts = turnIndex(session).turnStarts
  for (let position = starts.length - 1; position >= 0; position -= 1) {
    const start = starts[position]
    if (start !== undefined && start.turn === turn) return start.seq
  }
  return undefined
}

/**
 * The newest log seq of one turn that is not a record's own tool result — the
 * end of what a record written now would account for.
 *
 * `beforeSeq` bounds the scan, so a record written mid-turn accounts for what
 * preceded its own call and never claims the work that follows it.
 * @param session - session whose log to read.
 * @param records - every record of the session.
 * @param turn - the turn being recorded.
 * @param beforeSeq - inclusive upper bound on the scan; the log tail by default.
 * @returns the trailing trace seq, or `undefined` when the turn holds none.
 */
export function lastTraceSeqOfTurn(
  session: Session,
  records: readonly ContextRecord[],
  turn: number,
  beforeSeq?: number,
): number | undefined {
  const start = turnStartSeqOf(session, turn)
  if (start === undefined) return undefined
  const recordSeqs = new Set(records.map(record => record.seq))
  const events = session.events
  const from = beforeSeq === undefined ? events.length - 1 : Math.min(beforeSeq, events.length - 1)
  for (let seq = from; seq > start; seq -= 1) {
    if (recordSeqs.has(seq)) continue
    const event = events[seq]
    if (event === undefined) continue
    if (!hasTurn(event.data, turn)) continue
    return seq
  }
  return undefined
}

/**
 * The log seq of one tool call's own `tool/call` event.
 * @param session - session whose log to search.
 * @param callId - the execution's call identity.
 * @returns the event seq, or `undefined` when the log holds no such call.
 */
export function callEventSeqOf(session: Session, callId: string): number | undefined {
  const events = session.events
  for (let seq = events.length - 1; seq >= 0; seq -= 1) {
    const event = events[seq]
    if (event === undefined || event.type !== 'tool/call') continue
    if (event.data.callId === callId) return seq
  }
  return undefined
}

/** Whether one event payload names a turn. */
function hasTurn(data: unknown, turn: number): boolean {
  if (data === null || typeof data !== 'object') return false
  const record = data as Record<string, unknown>
  return record['turn'] === turn
}
