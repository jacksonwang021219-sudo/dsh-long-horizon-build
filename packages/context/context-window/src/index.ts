/**
 * Sliding-window context policy, carried by the harness rather than by the
 * agent's habits.
 *
 * The model-visible surface keeps the most recent user REQUESTS verbatim,
 * together with the summary the harness recorded for each of them; the
 * execution traces that answered them are replaced by those summaries. A
 * dedicated note keeper — a FORK of the main agent, so its request reuses the
 * same prefix cache — writes the notes, so the main agent never has to think
 * about recording and a turn's context bound does not depend on the model
 * choosing to cooperate.
 *
 * @module @deepseek-ai/dsh-context-window
 */

import z from '@deepseek-ai/schemastery'
import type { Context } from '@deepseek-ai/cordis'
import { assembleContextFor } from '@deepseek-ai/dsh-agent'
import type { Agent, PreStepDecision } from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { Message, UserMessage } from '@deepseek-ai/dsh-llm'
// Type-only: resolves `ctx.tokenMeter`, `ctx.tools`, `ctx.systemPrompt`, `ctx.llm`,
// and merges the `compaction/prune` shadow-price event.
import type {} from '@deepseek-ai/dsh-token-meter'
import type {} from '@deepseek-ai/dsh-tools'
import { renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import type {} from '@deepseek-ai/dsh-system-prompt'
import type { Session } from '@deepseek-ai/dsh-session'
import { resolveContextWindowConfig } from './config.ts'
import type { ContextWindowConfig, ResolvedContextWindowConfig } from './config.ts'
import { readIndex, readTaskNote, renderNotesContext, renderWindowSummary, writeRecording } from './notes.ts'
import type { IndexTask } from './notes.ts'
import { renderUnrecordedBlock } from './records.ts'
import { renderHistoryDigest } from './records.ts'
import type { DigestEntry } from './records.ts'
import { runRecorder } from './recorder.ts'
import { isChildSession, stateFor } from './state.ts'
import type { StoredNoteRef } from './state.ts'
import { registerWindowTools } from './tools.ts'
import { WINDOW_POLICY_ORDER, WINDOW_POLICY_SECTION, windowPolicyText } from './prompt.ts'
import {
  WINDOW_SOURCE_PLUGIN,
  foldPlanForObligation,
  humanPromptText,
  planWindow,
  turnStartSeqOf,
} from './window.ts'
import type { FoldPlan } from './window.ts'

export {
  DEFAULT_FOLD_TRIGGER_TOKENS,
  DEFAULT_RETAIN_TURN_TOKENS,
  DEFAULT_RETAIN_USER_TURNS,
  resolveContextWindowConfig,
} from './config.ts'
export type { ContextWindowConfig, ResolvedContextWindowConfig } from './config.ts'
export { NOTE_FIX_TOOL_NAME, NOTE_TOOL_NAME, LOG_TOOL_NAME, registerWindowTools } from './tools.ts'
export { WINDOW_POLICY_ORDER, WINDOW_POLICY_SECTION, windowPolicyText } from './prompt.ts'
export {
  WINDOW_SOURCE_PLUGIN,
  contentText,
  currentTurnOf,
  humanPromptText,
  isHumanPrompt,
  isWindowBlock,
  planWindow,
} from './window.ts'
export type { FoldPlan, RecordObligation, WindowDecision, WindowShadowSpan } from './window.ts'
export type {
  CorrectionRequest,
  IndexNote,
  NoteIndex,
  ReusableExperience,
  TaskNote,
  WindowSummary,
} from './notes.ts'

/** Row configuration schema; every field is optional and validated on resolve. */
const Config: z<ContextWindowConfig> = z.object({
  retainUserTurns: z.number().step(1).min(1),
  retainTurnTokens: z.number().step(1).min(1),
  foldTriggerTokens: z.number().step(1).min(1),
  digestMaxTokens: z.number().step(1).min(1),
  recordMaxChars: z.number().step(1).min(1),
  logReadMaxChars: z.number().step(1).min(1),
  recordReminders: z.number().step(1).min(0),
  recordMode: z.const('recorder'),
  policy: z.union([z.const('full'), z.const('minimal')]),
  recorderMaxTokens: z.number().step(1).min(1),
  recorderView: z.union([z.const('aligned'), z.const('span')]),
  recorderDedup: z.boolean(),
  notesRoot: z.string(),
  recorderRetries: z.number().step(1).min(1),
  recorderTimeoutMs: z.number().step(1).min(1),
  recorderWaitMs: z.number().step(1).min(1),
  recorderCircuitBreaker: z.number().step(1).min(1),
})

/** The span of one turn's own nodes on the current surface. */
interface TurnSpan {
  readonly turn: number
  readonly start: number
  readonly end: number
}

/**
 * The span one turn owns on the visible surface: its first node through the
 * newest node after it.
 * @param session - session whose surface to measure.
 * @param turn - the turn to bound.
 * @returns the span, or `undefined` when the turn owns nothing visible.
 */
function turnSpan(session: Session, turn: number): TurnSpan | undefined {
  const opening = turnStartSeqOf(session, turn)
  if (opening === undefined) return undefined
  let start: number | undefined
  let end: number | undefined
  for (const seq of session.surface.nodes) {
    if (seq <= opening) continue
    if (start === undefined) start = seq
    end = seq
  }
  return start === undefined || end === undefined ? undefined : { turn, start, end }
}

/**
 * The messages one span owns, for the recorder's `span` view.
 *
 * The span view replays only the recorded turn, so the request diverges from
 * the conversation at its first message and the provider cannot serve the
 * prompt from the cache the conversation warmed.
 * @param session - session whose surface to read.
 * @param start - inclusive first log seq of the span.
 * @param end - inclusive last log seq of the span.
 * @returns the span's messages, in surface order.
 */
function spanMessages(session: Session, start: number, end: number): Message[] {
  const messages: Message[] = []
  for (const seq of session.surface.nodes) {
    if (seq < start || seq > end) continue
    const event = session.events[seq]
    if (event === undefined) continue
    const message = session.deriveEventMessage(event)
    if (message !== null) messages.push(message)
  }
  return messages
}

/** The note already stored for one turn, when there is one. */
async function noteForTurn(
  config: ResolvedContextWindowConfig,
  session: Session,
  turn: number,
): Promise<StoredNoteRef | undefined> {
  const state = stateFor(session)
  const cached = state.notes.get(turn)
  if (cached !== undefined) return cached
  const index = await readIndex(config.notesRoot, session)
  const entry = index.tasks.find(task => task.turn === turn)
  if (entry === undefined) return undefined
  const note = await readTaskNote(config.notesRoot, session, entry.id)
  if (note === undefined) return undefined
  const ref: StoredNoteRef = { noteId: entry.id, summary: note.windowSummary, start: entry.start, end: entry.end }
  state.notes.set(turn, ref)
  return ref
}

/** Record one turn with the note keeper, store it, and remember the reference. */
async function recordTurn(
  ctx: Context,
  config: ResolvedContextWindowConfig,
  agent: Agent,
  span: TurnSpan,
  midTurn: boolean,
  signal: AbortSignal,
): Promise<StoredNoteRef | null> {
  const session = agent.session
  const index = await readIndex(config.notesRoot, session)
  const assembly = await ctx.systemPrompt.assemble(assembleContextFor(agent, signal))
  const prefix = {
    system: renderPrompt(assembly),
    tools: assembly.tools,
    messages: config.recorderView === 'aligned'
      ? session.deriveMessages()
      : spanMessages(session, span.start, span.end),
  }
  const outcome = await runRecorder(ctx, agent, prefix, {
    turn: span.turn,
    start: span.start,
    end: span.end,
    midTurn,
  }, renderNotesContext(index), config.notesRoot, {
    attempts: config.recorderRetries,
    maxTokens: config.recorderMaxTokens,
    dedup: config.recorderDedup,
  }, signal)
  if (outcome === null) return null
  const recording = outcome.recording
  const stored = await writeRecording(config.notesRoot, session, {
    turn: span.turn,
    start: span.start,
    end: span.end,
    title: recording.title,
    windowSummary: recording.windowSummary,
    body: recording.body,
    openItems: recording.openItems,
    reusableExperience: recording.reusableExperience,
    topics: recording.topics,
    types: recording.types,
    correctionVerdicts: recording.correctionVerdicts,
  })
  const ref: StoredNoteRef = {
    noteId: stored.noteId,
    summary: recording.windowSummary,
    start: span.start,
    end: span.end,
  }
  stateFor(session).notes.set(span.turn, ref)
  ctx.logger.info(
    `context-window: stored ${stored.noteId} for turn ${String(span.turn)} `
    + `(log ${String(span.start)}-${String(span.end)}${stored.changedIndex ? ', index updated' : ''})`,
  )
  return ref
}

/**
 * The folded-history line one stored note contributes.
 *
 * The conclusions are what makes the digest usable for dedup: a title alone
 * tells a later recording nothing about what has already been concluded.
 * @param task - the index entry of the note.
 * @returns the digest entry for that note.
 */
export function digestEntryOf(task: IndexTask): DigestEntry {
  return {
    turn: task.turn,
    start: task.start,
    end: task.end,
    summary: task.conclusions.length === 0
      ? `${task.id}: ${task.title}`
      : `${task.id}: ${task.title}\nConclusions: ${task.conclusions}`,
  }
}

/** Rebuild the folded-history digest from the notes stored so far. */
async function digestPlan(
  config: ResolvedContextWindowConfig,
  session: Session,
  plan: FoldPlan,
  estimate: (text: string) => number,
): Promise<FoldPlan> {
  const index = await readIndex(config.notesRoot, session)
  const entries: DigestEntry[] = index.tasks.map(task => digestEntryOf(task))
  if (entries.length === 0) return plan
  return { ...plan, text: renderHistoryDigest(entries, config.digestMaxTokens, estimate) }
}

/** Await an in-flight recording, bounded by the configured wait budget. */
async function settleRecording(
  ctx: Context,
  config: ResolvedContextWindowConfig,
  session: Session,
): Promise<void> {
  const state = stateFor(session)
  const handle = state.recording
  if (handle === undefined) return
  const outcome = await Promise.race([
    handle.done,
    new Promise<'timeout'>((resolve) => {
      setTimeout(() => { resolve('timeout') }, config.recorderWaitMs)
    }),
  ])
  state.recording = undefined
  if (outcome === 'timeout') {
    ctx.logger.warn(
      `context-window: recording of turn ${String(handle.turn)} is still running after `
      + `${String(Math.round(config.recorderWaitMs / 1000))}s; sending this request with the trace still visible`,
    )
    return
  }
  if (outcome === null) ctx.logger.warn(`context-window: turn ${String(handle.turn)} could not be recorded`)
}

/** Apply one fold: shared shadow-price event, then the replacement node. */
function applyFold(ctx: Context, session: Session, plan: FoldPlan): void {
  const message = createUserMessage({
    content: [{ type: 'text', text: plan.text }],
    source: { kind: 'plugin', plugin: WINDOW_SOURCE_PLUGIN },
  })
  const replacementTokens = ctx.tokenMeter.estimateMessage(message)
  if (replacementTokens >= plan.span.shadowedTokens) {
    ctx.logger.info(
      `context-window: the ${plan.reason} block (~${String(replacementTokens)} tokens) would not shrink `
      + `the span (~${String(plan.span.shadowedTokens)} tokens); keeping the full surface`,
    )
    return
  }
  session.append('compaction/prune', {
    shadowedRange: { start: plan.span.start, end: plan.span.end },
    shadowedSeqs: [...plan.span.shadowedSeqs],
    shadowedTokenCount: plan.span.shadowedTokens,
  })
  session.append('user/message', message, {
    surfaceOp: { op: 'replace', start: plan.span.start, end: plan.span.end },
    sourceEventSeqs: [...plan.span.shadowedSeqs],
  })
  stateFor(session).folds += 1
  ctx.logger.info(
    `context-window: ${plan.reason} folded ${String(plan.span.shadowedSeqs.length)} node(s) `
    + `(~${String(plan.span.shadowedTokens)} -> ~${String(replacementTokens)} tokens)`,
  )
}

/** Trip the breaker: stop folding and recording for this session, visibly. */
function freeze(ctx: Context, session: Session, reason: string): void {
  const state = stateFor(session)
  state.frozen = true
  state.recording = undefined
  ctx.logger.warn(`context-window: ${reason}; this session now keeps its full history and relies on compaction`)
  if (state.announced) return
  state.announced = true
  const notice = createUserMessage({
    content: [{
      type: 'text',
      text: [
        '<window-notice>',
        'Context recording has failed repeatedly in this session, so the harness has stopped folding history and will keep the full conversation visible from here on. Nothing is lost; carry on with the task.',
        '</window-notice>',
      ].join('\n'),
    }],
    source: { kind: 'plugin', plugin: WINDOW_SOURCE_PLUGIN, form: 'notice', summary: 'Context recording disabled for this session' },
  })
  session.append('user/message', notice, { surfaceOp: 'append' })
}

/** One window pass, run at every pre-step of a top-level session. */
async function windowPass(
  ctx: Context,
  config: ResolvedContextWindowConfig,
  agent: Agent,
  messages: readonly UserMessage[],
  signal: AbortSignal,
): Promise<void> {
  const session = agent.session
  const state = stateFor(session)
  if (state.frozen || signal.aborted) return
  const estimate = (text: string): number => ctx.tokenMeter.estimateMessage(createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: WINDOW_SOURCE_PLUGIN },
  }))
  const priced = new Map<number, number>()
  for (const node of ctx.tokenMeter.measure(session).nodes) priced.set(node.seq, node.tokens)
  const decision = planWindow(session, {
    config,
    records: [],
    priced,
    estimate,
    incomingPrompt: humanPromptText(messages) !== undefined,
    pending: state.pending,
    recordMode: config.recordMode,
  })
  if (decision.kind === 'none') return

  if (decision.kind === 'record-first') {
    // 400K inside one request: serial by definition, because the turn cannot
    // continue until its context fits again.
    const plan = foldPlanForObligation(session, decision.obligation, priced, estimate)
    if (plan === null) return
    const span = turnSpan(session, decision.obligation.turn)
    const recorded = span === undefined
      ? null
      : await recordTurn(ctx, config, agent, span, true, signal)
    if (recorded === null) {
      state.failures += 1
      if (state.failures >= config.recorderCircuitBreaker) {
        freeze(ctx, session, `turn ${String(decision.obligation.turn)} could not be recorded`)
        return
      }
      applyFold(ctx, session, plan)
      return
    }
    state.failures = 0
    state.pending = undefined
    state.pendingAttempts = 0
    applyFold(ctx, session, {
      ...plan,
      // The block must cite the range its note was written for: the planner's
      // span is where the fold started, and an in-turn recording rewrites the
      // turn's whole span, so the two can differ.
      text: renderWindowSummary(decision.obligation.turn, recorded.start, recorded.end, recorded.summary, recorded.noteId),
    })
    return
  }

  if (decision.plan.turn === undefined) {
    applyFold(ctx, session, await digestPlan(config, session, decision.plan, estimate))
    return
  }
  // A turn may only be folded once it is recorded. If the background recording
  // has not landed (still running, or it failed), record it here and now; a
  // fold that could not be recorded cites the log range instead of pretending
  // to carry a note.
  const turn = decision.plan.turn
  const note = await noteForTurn(config, session, turn)
    ?? await (async () => {
      const span = turnSpan(session, turn)
      return span === undefined ? null : recordTurn(ctx, config, agent, span, false, signal)
    })()
  if (note === null) {
    state.failures += 1
    if (state.failures >= config.recorderCircuitBreaker) {
      freeze(ctx, session, `turn ${String(turn)} could not be recorded`)
      return
    }
    const span = decision.plan.span
    applyFold(ctx, session, {
      ...decision.plan,
      text: renderUnrecordedBlock(turn, span.start, span.end, span.shadowedTokens),
    })
    return
  }
  state.failures = 0
  applyFold(ctx, session, {
    ...decision.plan,
    text: renderWindowSummary(turn, note.start, note.end, note.summary, note.noteId),
  })
}

/** Bind the policy to one agent scope: a newly mounting session gets its own row. */
function apply(ctx: Context, config: ContextWindowConfig = {}): void {
  const resolved = resolveContextWindowConfig(config)
  registerWindowTools(ctx, resolved)
  ctx.systemPrompt.section({
    name: WINDOW_POLICY_SECTION,
    order: WINDOW_POLICY_ORDER,
    // A delegated child — including the note keeper itself — inherits this
    // composition; the protocol is written for the user-facing agent.
    text: (assembly) => {
      const agent = assembly.agent
      if (agent === undefined) return ''
      return isChildSession(agent.session) ? '' : windowPolicyText(resolved)
    },
  })

  ctx.on('agent/pre-step', async (
    { agent, messages, turn, step, signal },
    next,
  ): Promise<PreStepDecision> => {
    const decision = await next()
    if (decision.kind === 'reject') return decision
    const session = agent.session
    if (isChildSession(session)) return decision
    const state = stateFor(session)
    if (humanPromptText(messages) !== undefined) state.humanTurns.add(turn)
    try {
      // The pseudo-async handshake: a query that arrives while the previous
      // turn is still being recorded waits here for it, so the request is built
      // against the folded window it was promised.
      await settleRecording(ctx, resolved, session)
      if (state.frozen) return decision
      await windowPass(ctx, resolved, agent, decision.messages, signal)
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : String(error)
      ctx.logger.warn(`context-window: window pass failed on turn ${String(turn)} step ${String(step)} (${detail})`)
    }
    return decision
  })

  // End of a user turn: start the recording in the background. The reply the
  // user sees is never blocked on it; the next request waits for it above.
  ctx.on('agent/turn-stopping', ({ agent, turn, signal }) => {
    const session = agent.session
    if (signal.aborted || isChildSession(session)) return
    const state = stateFor(session)
    if (state.frozen || state.recording !== undefined || state.notes.has(turn)) return
    if (!state.humanTurns.has(turn)) return
    const span = turnSpan(session, turn)
    if (span === undefined) return
    const controller = new AbortController()
    state.recording = {
      turn,
      start: span.start,
      end: span.end,
      done: recordTurn(ctx, resolved, agent, span, false, controller.signal)
        .then((outcome) => {
          const current = stateFor(session)
          if (outcome === null) current.failures += 1
          else current.failures = 0
          return outcome
        })
        .catch((error: unknown) => {
          const detail = error instanceof Error ? error.message : String(error)
          ctx.logger.warn(`context-window: recording turn ${String(turn)} failed (${detail})`)
          stateFor(session).failures += 1
          return null
        }),
    }
  })
}

/** The sliding-window plugin row: its schema, its service dependencies, and its registrations. */
export default {
  name: 'dsh-context-window',
  inject: ['tokenMeter', 'systemPrompt', 'tools', 'llm'],
  Config,
  apply,
}
