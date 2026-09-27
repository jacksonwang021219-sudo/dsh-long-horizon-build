/**
 * Window planning against synthetic session logs: what leaves the surface,
 * what stops a fold, and when the window must ask for a record first.
 */

import { describe, expect, it } from 'vitest'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { CallId, createMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import type { ToolResultMessage } from '@deepseek-ai/dsh-llm'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { resolveContextWindowConfig } from '../src/config.ts'
import { digestEntryOf } from '../src/index.ts'
import {
  CONCLUSION_DIGEST_CHARS,
  FOLDED_RECORD_PREFIX,
  appendRecorderUsage,
  readIndex,
  renderNotesContext,
  renderWindowSummary,
  sessionNotesDir,
  writeRecording,
} from '../src/notes.ts'
import type { NoteDraft } from '../src/notes.ts'
import { recordingInstruction } from '../src/recorder.ts'
import { RECORD_META_KIND, sessionRecords } from '../src/records.ts'
import { currentTurnOf, foldPlanForObligation, planWindow, turnStartSeqOf } from '../src/window.ts'

const MODEL = 'test-model'

/** A context carrying only the token meter the planner prices with. */
function createContext(): Context {
  const ctx = new Context()
  void new TokenMeter(ctx)
  return ctx
}

/** Estimated tokens of one plain text under the meter's estimator. */
function estimate(ctx: Context, text: string): number {
  return ctx.tokenMeter.estimateMessage(createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: 'test' },
  }))
}

/** Append one assistant text step to an open turn. */
function assistantStep(session: Session, turn: number, step: number, text: string): void {
  session.append('step/start', { turn, step })
  session.append('assistant/message', {
    turn,
    step,
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'text', text }],
      source: { kind: 'model', provider: MODEL, model: MODEL },
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step })
}

/** Append the durable record a `record_turn` call leaves in the log. */
function recordStep(
  session: Session,
  turn: number,
  step: number,
  summary: string,
  range: { start: number; end: number },
): void {
  session.append('step/start', { turn, step })
  const callId = CallId(`record-${String(turn)}-${String(step)}`)
  // The loop logs the call inside its assistant message and the pairing table
  // reads calls from there, so the fixture mirrors that order.
  session.append('assistant/message', {
    turn,
    step,
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'tool-call', id: callId, name: 'record_turn', arguments: '{}' }],
      source: { kind: 'model', provider: MODEL, model: MODEL },
    }),
  }, { surfaceOp: 'append' })
  session.append('tool/call', { turn, step, callId, name: 'record_turn', arguments: '{}' })
  // The message is normalized through JSON because `createToolResultMessage`
  // carries an explicit `undefined` the durable log refuses; the runtime
  // materializes the same result without it.
  const message = JSON.parse(JSON.stringify(
    createToolResultMessage({ callId, isError: false, content: [{ type: 'text', text: 'Recorded.' }] }),
  )) as ToolResultMessage
  session.append('tool/result', {
    turn,
    step,
    message,
    meta: {
      kind: RECORD_META_KIND,
      turn,
      recordKind: 'turn',
      start: range.start,
      end: range.end,
      tokens: 10,
      time: 1,
      summary,
    },
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step })
}

/**
 * One closed user turn: request, one assistant step, then a record covering
 * the whole turn, followed by the final answer.
 */
function turn(session: Session, turnNumber: number, size: number): void {
  session.append('turn/start', { turn: turnNumber })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: `request ${String(turnNumber)}` }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  assistantStep(session, turnNumber, 1, 'trace '.repeat(size))
  const lastSeq = session.events.length - 1
  recordStep(session, turnNumber, 2, `record for turn ${String(turnNumber)}`, {
    start: turnStartSeqOf(session, turnNumber) ?? 0,
    end: lastSeq,
  })
  assistantStep(session, turnNumber, 3, `final answer ${String(turnNumber)}`)
  session.append('turn/end', { turn: turnNumber, reason: { kind: 'completed' } })
}

/** Price every current surface node through the meter. */
function priced(ctx: Context, session: Session): Map<number, number> {
  const map = new Map<number, number>()
  for (const node of ctx.tokenMeter.measure(session).nodes) map.set(node.seq, node.tokens)
  return map
}

/** Plan one window pass under a given policy. */
function plan(
  ctx: Context,
  session: Session,
  config: Parameters<typeof planWindow>[1]['config'],
  incomingPrompt = true,
): ReturnType<typeof planWindow> {
  return planWindow(session, {
    config,
    records: sessionRecords(session),
    priced: priced(ctx, session),
    estimate: text => estimate(ctx, text),
    incomingPrompt,
    recordMode: 'recorder',
  })
}

describe('resolveContextWindowConfig', () => {
  it('rejects a trigger that does not leave room above the retained window', () => {
    expect(() => resolveContextWindowConfig({ retainTurnTokens: 200_000, foldTriggerTokens: 200_000 }))
      .toThrow(/must exceed/)
  })

  it('rejects unknown keys', () => {
    expect(() => resolveContextWindowConfig({ assemblerTools: ['read'] } as never)).toThrow(/unknown key/)
  })

  it('defaults to five requests, 200K retained, 400K trigger', () => {
    const resolved = resolveContextWindowConfig()
    expect(resolved.retainUserTurns).toBe(5)
    expect(resolved.retainTurnTokens).toBe(200_000)
    expect(resolved.foldTriggerTokens).toBe(400_000)
  })
})

describe('sessionRecords', () => {
  it('extracts records incrementally from the durable log', () => {
    const session = Session.create(SessionId('records'))
    turn(session, 1, 2)
    const first = sessionRecords(session)
    expect(first).toHaveLength(1)
    expect(first[0]?.turn).toBe(1)
    expect(first[0]?.summary).toBe('record for turn 1')
    turn(session, 2, 2)
    const second = sessionRecords(session)
    expect(second.map(record => record.turn)).toEqual([1, 2])
  })
})

describe('planWindow', () => {
  const config = resolveContextWindowConfig({ retainUserTurns: 2, digestMaxTokens: 4096 })
  const wide = resolveContextWindowConfig({ retainUserTurns: 5, digestMaxTokens: 4096 })

  it('leaves a session inside the window alone', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('inside-window'))
    turn(session, 1, 4)
    expect(plan(ctx, session, config).kind).toBe('none')
  })

  it('folds a completed turn trace when the window writes the record itself', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('trace-fold'))
    turn(session, 1, 400)
    turn(session, 2, 400)
    const decision = plan(ctx, session, wide)
    expect(decision.kind).toBe('fold')
    if (decision.kind !== 'fold') return
    expect(decision.plan.reason).toBe('turn-trajectory')
    // The fold replaces only turn 1's trace, never its request.
    expect(decision.plan.span.shadowedSeqs.length).toBeGreaterThan(0)
  })

  it('keeps a span unfolded until a record covers it when the agent is the writer', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('agent-written'))
    turn(session, 1, 400)
    turn(session, 2, 400)
    const decision = planWindow(session, {
      config: wide,
      records: [],
      priced: priced(ctx, session),
      estimate: text => estimate(ctx, text),
      incomingPrompt: true,
      recordMode: 'agent',
    })
    expect(decision.kind).toBe('none')
  })

  it('folds a completed turn in recorder mode before any record exists', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('recorder-mode'))
    turn(session, 1, 400)
    turn(session, 2, 400)
    const decision = planWindow(session, {
      config: wide,
      records: [],
      priced: priced(ctx, session),
      estimate: text => estimate(ctx, text),
      incomingPrompt: true,
      recordMode: 'recorder',
    })
    expect(decision.kind).toBe('fold')
    if (decision.kind !== 'fold') return
    expect(decision.plan.turn).toBe(1)
    expect(decision.plan.text).toContain('source="recorder"')
  })

  it('folds turns older than the retained requests into one digest', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('digest'))
    turn(session, 1, 400)
    turn(session, 2, 400)
    turn(session, 3, 400)
    const decision = plan(ctx, session, config)
    expect(decision.kind).toBe('fold')
    if (decision.kind !== 'fold') return
    expect(decision.plan.reason).toBe('history-digest')
    expect(decision.plan.text).toContain('<folded-history>')
    expect(decision.plan.text).toContain('record for turn 1')
  })

  it('asks for a record before folding a turn in progress past the trigger', () => {
    const ctx = createContext()
    const session = Session.create(SessionId('intra-turn'))
    const tiny = resolveContextWindowConfig({
      retainUserTurns: 5,
      retainTurnTokens: 100,
      foldTriggerTokens: 200,
    })
    session.append('turn/start', { turn: 1 })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'one long request' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    for (let step = 1; step <= 6; step += 1) assistantStep(session, 1, step, 'trace '.repeat(200))
    const decision = plan(ctx, session, tiny)
    expect(decision.kind).toBe('record-first')
    if (decision.kind !== 'record-first') return
    expect(decision.obligation.turn).toBe(1)
    expect(decision.obligation.end).toBeGreaterThan(decision.obligation.start)
    expect(currentTurnOf(session)).toBe(1)
    // The obligation is bounded by the retained window the fold must reach.
    expect(decision.turnTokens).toBeGreaterThanOrEqual(tiny.foldTriggerTokens)

    // Once the record exists for that range, the same pass folds instead. The
    // plan reuses the boundary it asked for, so the record still matches even
    // though the record step itself added nodes.
    recordStep(session, 1, 7, 'partial record', {
      start: decision.obligation.start,
      end: decision.obligation.end,
    })
    // …and when the agent never records, the same boundary is folded anyway,
    // with a block that cites the uncovered range instead of a summary.
    const fallback = foldPlanForObligation(
      session,
      decision.obligation,
      priced(ctx, session),
      text => estimate(ctx, text),
    )
    expect(fallback).not.toBeNull()
    expect(fallback?.text).toContain('<unrecorded-trace')
    expect(fallback?.text).toContain(String(decision.obligation.start))
    expect(fallback?.span.end).toBe(decision.obligation.end)

    const after = planWindow(session, {
      config: tiny,
      records: sessionRecords(session),
      priced: priced(ctx, session),
      estimate: text => estimate(ctx, text),
      incomingPrompt: true,
      pending: { turn: decision.obligation.turn, end: decision.obligation.end },
      recordMode: 'recorder',
    })
    expect(after.kind).toBe('fold')
    if (after.kind !== 'fold') return
    expect(after.plan.reason).toBe('intra-turn')
    expect(after.plan.text).toContain('partial record')
  })
})

describe('note index', () => {
  /** One recording draft whose conclusions are the part under test. */
  function draft(turnNumber: number, summary: NoteDraft['windowSummary']): NoteDraft {
    return {
      turn: turnNumber,
      start: turnNumber * 10,
      end: turnNumber * 10 + 5,
      title: `turn ${String(turnNumber)}`,
      windowSummary: summary,
      body: 'body',
      openItems: '',
      reusableExperience: [],
      topics: [],
      types: [],
      correctionVerdicts: [],
    }
  }

  it('keeps each note\'s conclusions in the index, capped to the entry budget', async () => {
    const session = Session.create(SessionId('index-conclusions'))
    const root = await mkdtemp(join(tmpdir(), 'dsh-notes-'))
    try {
      await writeRecording(root, session, draft(1, {
        taskDefinition: 'build the thing',
        strategyAndProgress: 'tried once',
        keyConclusions: 'x'.repeat(CONCLUSION_DIGEST_CHARS + 50),
        deliveryAndAcceptance: 'none yet',
      }))
      const index = await readIndex(root, session)
      expect(index.tasks).toHaveLength(1)
      expect(index.tasks[0]?.conclusions).toHaveLength(CONCLUSION_DIGEST_CHARS)
      await expect(readFile(join(sessionNotesDir(root, session), 'tasks', 'task-0001.json'), 'utf8'))
        .resolves.toContain('x'.repeat(CONCLUSION_DIGEST_CHARS + 50))
      // The recorder's own view of what is already recorded carries it too.
      expect(renderNotesContext(index)).toContain('conclusions already recorded:')
      // …and so does the folded-history line an expired turn leaves behind.
      const entry = digestEntryOf(index.tasks[0]!)
      expect(entry.summary).toContain('Conclusions:')
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })

  it('records the provider accounting of every recorder call', async () => {
    const session = Session.create(SessionId('index-usage'))
    const root = await mkdtemp(join(tmpdir(), 'dsh-notes-'))
    try {
      await appendRecorderUsage(root, session, {
        turn: 1,
        attempt: 1,
        accepted: true,
        provider: 'deepseek',
        model: MODEL,
        durationMs: 1200,
        inputTokens: 10,
        outputTokens: 20,
        cacheReadTokens: 3000,
        cacheWriteTokens: 0,
        reasoningTokens: 5,
      })
      const lines = (await readFile(join(sessionNotesDir(root, session), 'recorder-usage.jsonl'), 'utf8')).trim().split('\n')
      expect(lines).toHaveLength(1)
      expect(JSON.parse(lines[0] ?? '{}')).toMatchObject({ turn: 1, cacheReadTokens: 3000, accepted: true })
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
})

describe('recordingInstruction', () => {
  const task = { turn: 3, start: 10, end: 40, midTurn: false }

  it('forbids repeating recorded conclusions only when dedup is on', () => {
    expect(recordingInstruction(task, '<existing-notes></existing-notes>', true)).toContain('HARD RULE')
    expect(recordingInstruction(task, '<existing-notes></existing-notes>', false)).not.toContain('HARD RULE')
    expect(recordingInstruction(task, '<existing-notes></existing-notes>', true)).toContain('seq 10 to 40')
  })

  it('carries the retry reason only on a retry', () => {
    expect(recordingInstruction(task, '', true)).not.toContain('not accepted')
    expect(recordingInstruction(task, '', true, 'invalid JSON')).toContain('invalid JSON')
  })

  it('forbids naming who performed the work', () => {
    const contract = recordingInstruction(task, '', true)
    expect(contract).toContain('Write no third person')
    expect(contract).toContain('never who performed it')
  })

  it('forbids a field that only points at an earlier note', () => {
    expect(recordingInstruction(task, '', true)).toContain('bare pointer at something already recorded')
  })
})

describe('renderWindowSummary', () => {
  const summary = {
    taskDefinition: 'Create report.md',
    strategyAndProgress: 'Written in one pass.',
    keyConclusions: 'report.md holds the three findings.',
    deliveryAndAcceptance: 'report.md exists, 1204 bytes, unread.',
  }

  it('opens the record with the fixed line and keeps the note identity', () => {
    const text = renderWindowSummary(4, 100, 180, summary, 'task-0002')
    expect(text.startsWith(`<turn-summary turn="4" note="task-0002" log-range="100-180">\n`)).toBe(true)
    expect(text).toContain(`\n${FOLDED_RECORD_PREFIX}\n`)
    expect(text.indexOf(FOLDED_RECORD_PREFIX)).toBeLessThan(text.indexOf('**Task definition.** Create report.md'))
    expect(text).toContain('session_log')
  })

  it('states the fixed line once per record', () => {
    const text = renderWindowSummary(4, 100, 180, summary, 'task-0002')
    expect(text.split(FOLDED_RECORD_PREFIX)).toHaveLength(2)
  })
})
