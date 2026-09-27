/**
 * The note keeper: ONE model call that turns a turn's execution trace into a
 * short page of conclusions, plus the harness-side reading of its reply.
 *
 * One call, no tools, no agent loop: the request replays the trace and appends
 * the contract, and the harness applies the returned patch to the note store.
 * The note is deliberately small — four fields of conclusions and, at most, a
 * couple of reusable lessons — because its whole purpose is to be read whole by
 * the next turn.
 *
 * @module @deepseek-ai/dsh-context-window/recorder
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, Message, TokenUsage, ToolSchema } from '@deepseek-ai/dsh-llm'
import { appendRecorderLog, appendRecorderUsage, salvageFields } from './notes.ts'
import type { RecorderUsageEntry, ReusableExperience, VerificationLevel, WindowSummary } from './notes.ts'
import { coerceWindowSummary } from './notes.ts'
import { contentTextOf } from './text.ts'
import { WINDOW_SOURCE_PLUGIN } from './window.ts'

/** Purpose stamped on the auxiliary recording request. */
const RECORDER_PURPOSE = 'context-record'

/** One correction the main agent raised, awaiting the recorder's audit. */
export interface PendingCorrection {
  /** The note the agent believes is wrong. */
  readonly noteId: string
  /** What is wrong, specifically. */
  readonly problem: string
  /** How the agent thinks it should read instead. */
  readonly fix: string
  /** The evidence the agent offers. */
  readonly evidence: string
}

/** What one recording run is asked to cover. */
export interface RecordingTask {
  /** The turn being recorded. */
  readonly turn: number
  /** Inclusive first log seq the record must account for. */
  readonly start: number
  /** Inclusive last log seq the record must account for. */
  readonly end: number
  /** Whether this recording interrupts the turn in progress (the 400K case). */
  readonly midTurn: boolean
}

/** One parsed recording: the note plus the index changes it asks for. */
export interface Recording {
  /** The four window fields; the note's substance. */
  readonly windowSummary: WindowSummary
  /** Short title for the index. */
  readonly title: string
  /** The note page, rendered from the four fields. */
  readonly body: string
  /** Extra open items, when the recorder volunteered them. */
  readonly openItems: string
  /** Lessons worth reusing, possibly empty. */
  readonly reusableExperience: readonly ReusableExperience[]
  /** Topic entries to create or update. */
  readonly topics: readonly { name: string; description: string }[]
  /** Type entries to create or update. */
  readonly types: readonly { name: string; description: string }[]
  /** Verdicts on the main agent's corrections. */
  readonly correctionVerdicts: readonly {
    readonly noteId: string
    readonly verdict: 'accepted' | 'rejected' | 'disputed'
    readonly reason: string
  }[]
}

/** One completed recording run. */
export interface RecorderOutcome {
  /** The parsed patch. */
  readonly recording: Recording
  /** Which attempt succeeded, 1-based. */
  readonly attempts: number
}

/** Retry budget and generation ceiling for one recording. */
export interface RecorderOptions {
  /** Attempts before the caller degrades. */
  readonly attempts: number
  /** Generation ceiling for one recording call, in tokens. */
  readonly maxTokens: number
  /** Whether the contract forbids repeating conclusions the notes already carry. */
  readonly dedup: boolean
}

/**
 * The recording contract, delivered as the FINAL user message.
 *
 * It is text, not a tool definition, on purpose: adding a tool would change the
 * request's tool block and invalidate the prefix cache the replayed trace
 * shares with the conversation. The note it asks for is small by design.
 * @param task - the span this recording must cover.
 * @param notesContext - the existing index, in full, so the patch can say "none".
 * @param dedup - whether to forbid repeating conclusions the notes already carry.
 * @param previousProblem - why the previous attempt was not accepted.
 * @returns the instruction text.
 */
export function recordingInstruction(
  task: RecordingTask,
  notesContext: string,
  dedup: boolean,
  previousProblem?: string,
): string {
  return [
    'You are the note keeper for this session. The messages above are the execution trace of ONE turn; the window replaces them with the note you write here.',
    '',
    'This is a single reply, not an agent task: there is NO tool available to you, and a tool call is not an acceptable reply. Do not continue the task. Read what is above and answer with the JSON patch below.',
    '',
    `Record turn ${String(task.turn)}, which covered log seq ${String(task.start)} to ${String(task.end)}.`,
    task.midTurn
      ? 'That turn is still running and is about to exceed its context budget: record what is settled so far and leave the rest out.'
      : 'The note lands in the window in place of the trace it replaces, so it is read as this session\'s own account of that turn: what the turn settled, carried forward to the next turn. Keep the session\'s language, name only what the note is about and never who performed it, and describe events by their object, path, command or result.',
    '',
    notesContext,
    '',
    'Reply with ONE JSON object and nothing else:',
    '',
    '```json',
    JSON.stringify({
      task: {
        title: 'short title, at most 12 words',
        task_definition: 'the request this turn answered',
        strategy_and_progress: 'what was attempted, and the state things are in now',
        key_conclusions: 'what is now known that a later turn must not have to rediscover',
        delivery_and_acceptance: 'what was produced, where it lives, and whether it was verified',
        reusable_experience: [{
          experience: 'a lesson a later session could reuse',
          verification: 'verified | partially-verified | hypothesis',
          conditions: 'the preconditions under which it holds',
          when_to_reuse: 'when it is worth reusing',
          evidence: 'log seqs or workspace facts it rests on',
        }],
      },
      index_patch: {
        topics: [{ name: 'topic name', change: 'create | update | none', description: 'the FULL new description when change is create or update; empty otherwise' }],
        types: [{ name: 'type name', change: 'create | update | none', description: 'the FULL new description when change is create or update; empty otherwise' }],
      },
      correction_verdicts: [{ note_id: 'task-0001', verdict: 'accepted | rejected | disputed', reason: 'why' }],
    }, null, 2),
    '```',
    '',
    'How to write it — this is the whole point of the note:',
    ...dedup ? ['- HARD RULE: any conclusion already covered by <existing-notes> or already visible in the window (an earlier <turn-summary>, an earlier note, a topic or type description) MUST NOT be repeated. Do not restate it, do not rephrase it, do not re-derive it. A field with nothing new for THIS turn is left empty — write "" for it. A note that repeats what is already recorded is worse than an empty note, because it buries the new facts.'] : [],
    '- CONCLUSIONS ONLY. Each field holds the conclusion, not the reasoning behind it, not a narrative of what was performed, and not an explanation of what the field means. State the fact and stop.',
    '- Keep the session\'s own terms. Name every file, path, symbol, command, identifier and number exactly as the trace names it, in the language the trace uses for it. Never paraphrase a name into a description of it.',
    '- Write no third person. Never write "the agent", "the assistant", "the note keeper" or "the session", and never state who performed the work. Every field must read the same whether the work was done or merely established.',
    '- Nothing in this note may be a bare pointer at something already recorded. "See an earlier note", "as established before", "unchanged from the previous turn" carry no information a later turn can act on: either state the fact, or leave the field empty.',
    '- Prefer the exact path, command, identifier or number over a description of it. Prefer a fact a later model can act on directly over one it must interpret.',
    '- Never paste logs, diffs, file contents or tool output. Never restate the request. Never explain what you are about to write.',
    '- Stay SHORT: the title plus the four fields together must stay under 2000 characters, and shorter is better. Do not treat 2000 as a target to fill — write the smallest note that still carries every conclusion a later turn needs.',
    '- `reusable_experience` may be an empty list, and is usually empty. At most 2 entries, one line each.',
    '- `index_patch`: for every entry in <existing-notes> decide `change`. Use "none" with an empty description when this turn does not change it — that is the normal case and it produces no diff. Use "create" for something new, "update" only when this turn genuinely changes the description, then give the FULL replacement description in one or two sentences.',
    '- `correction_verdicts` audits the correction requests listed in <existing-notes> against the trace above. Omit the list when there are none.',
    '- A reply that gets cut off cannot be read at all, so keep the whole object small.',
    ...previousProblem === undefined ? [] : [
      '',
      `Your previous reply was not accepted: ${previousProblem}. Reply again with the JSON object, shorter.`,
    ],
  ].join('\n')
}

/** The prefix a recording request replays, so it shares the main agent's cache. */
export interface RecordingPrefix {
  /** The assembled system prompt, exactly as the conversation sends it. */
  readonly system: string
  /** The tool schemas in catalog order, exactly as the conversation sends them. */
  readonly tools: readonly ToolSchema[]
  /** The model-visible window: recent requests, their summaries, and this turn. */
  readonly messages: readonly Message[]
}

/**
 * Run one recording: a single model call, retried on an unreadable reply.
 *
 * The request replays the conversation's own prefix — system, tools, and the
 * model-visible window — and appends the contract, so the provider reuses the
 * cache the conversation just warmed and the note keeper sees exactly what the
 * main agent sees (including this turn's trace, which is still on the surface).
 * Only the tail differs, and a prefix cache is unaffected by its tail.
 * @param ctx - context providing the LLM service.
 * @param agent - the agent whose route and session the call borrows.
 * @param prefix - the aligned system, tools, and messages to replay.
 * @param task - the span's coordinates, for the contract and the diagnostics.
 * @param notesContext - the existing index, rendered in full.
 * @param notesRoot - where the session's diagnostics and cost accounting are written.
 * @param options - retry budget, generation ceiling, and contract rules.
 * @param signal - the caller's cancellation signal.
 * @returns the parsed recording, or `null` when every attempt failed.
 */
export async function runRecorder(
  ctx: Context,
  agent: Agent,
  prefix: RecordingPrefix,
  task: RecordingTask,
  notesContext: string,
  notesRoot: string,
  options: RecorderOptions,
  signal: AbortSignal,
): Promise<RecorderOutcome | null> {
  const session = agent.session
  const routed = session.requestHeader()?.config
  const provider = routed?.provider ?? agent.options.provider ?? ''
  const model = routed?.model ?? agent.options.model ?? ''
  if (provider.length === 0 || model.length === 0) {
    await appendRecorderLog(notesRoot, session, `no provider/model for turn ${String(task.turn)}`)
    return null
  }
  await appendRecorderLog(
    notesRoot,
    session,
    `turn ${String(task.turn)}: recording seq ${String(task.start)}-${String(task.end)}; `
    + `${String(prefix.messages.length)} window message(s), ${String(prefix.tools.length)} tool(s)`,
  )
  let problem: string | undefined
  for (let attempt = 1; attempt <= Math.max(1, options.attempts); attempt += 1) {
    if (signal.aborted) return null
    const started = Date.now()
    const assembler = new BlockAssembler()
    const request: GenerateOptions = {
      provider,
      model,
      system: prefix.system,
      tools: [...prefix.tools],
      messages: [
        ...prefix.messages,
        createUserMessage({
          content: [{ type: 'text', text: recordingInstruction(task, notesContext, options.dedup, problem) }],
          source: { kind: 'plugin', plugin: WINDOW_SOURCE_PLUGIN },
        }),
      ],
      maxTokens: options.maxTokens,
      sessionId: session.id,
      purpose: RECORDER_PURPOSE,
      signal,
    }
    try {
      for await (const chunk of ctx.llm.stream(request)) assembler.push(chunk)
      const reply = contentTextOf(assembler.blocks())
      const parsed = parseRecording(reply, task)
      await appendRecorderUsage(notesRoot, session, {
        turn: task.turn,
        attempt,
        accepted: parsed !== null,
        provider,
        model,
        durationMs: Date.now() - started,
        ...countUsage(assembler.usage),
      })
      if (parsed === null) {
        const why = recordingProblem(reply) ?? 'the JSON carried no window fields'
        const cutOff = why.startsWith('invalid JSON') || why.startsWith('no braces')
        problem = cutOff
          ? `the reply was cut off or unparsable (${String(reply.length)} chars, ${why}). Answer shorter: under 2000 characters total.`
          : `the reply carried no usable patch (${String(reply.length)} chars, ${why}). Reply with the JSON object only: no tool call, no markup around it.`
        await appendRecorderLog(notesRoot, session, `attempt ${String(attempt)}: ${problem}`)
        continue
      }
      const counted = countUsage(assembler.usage)
      ctx.logger.info(
        `context-window: recorded turn ${String(task.turn)} in one call on attempt ${String(attempt)} `
        + `(${String(Math.round((Date.now() - started) / 1000))}s, ${String(counted.inputTokens)} input / `
        + `${String(counted.cacheReadTokens)} cached / ${String(counted.outputTokens)} output tokens)`,
      )
      return { recording: parsed, attempts: attempt }
    } catch (error: unknown) {
      problem = error instanceof Error ? error.message : String(error)
      await appendRecorderLog(notesRoot, session, `attempt ${String(attempt)} threw: ${problem}`)
    }
  }
  await appendRecorderLog(
    notesRoot,
    session,
    `turn ${String(task.turn)} GAVE UP after ${String(Math.max(1, options.attempts))} attempt(s); last problem: ${String(problem)}`,
  )
  return null
}

/**
 * Normalize a provider's usage report into the accounting fields the usage log
 * carries, so a call that reported nothing still leaves a priced row of zeros.
 * @param usage - the provider's report, when one arrived.
 * @returns the five token fields.
 */
function countUsage(usage: TokenUsage | undefined): Omit<RecorderUsageEntry, 'turn' | 'attempt' | 'accepted' | 'provider' | 'model' | 'durationMs'> {
  return {
    inputTokens: usage?.inputTokens ?? 0,
    outputTokens: usage?.outputTokens ?? 0,
    cacheReadTokens: usage?.cacheReadTokens ?? 0,
    cacheWriteTokens: usage?.cacheWriteTokens ?? 0,
    reasoningTokens: usage?.reasoningTokens ?? 0,
  }
}

/**
 * Parse one recorder reply, leniently.
 *
 * The four fields may sit directly on `task` (the contract) or under a
 * `window_summary` key (a reasonable variation); a reply that was cut off is
 * salvaged field by field rather than discarded.
 * @param text - the recorder's reply.
 * @param task - the span the harness asked for.
 * @returns the parsed recording, or `null` when nothing usable was found.
 */
export function parseRecording(text: string, task: RecordingTask): Recording | null {
  const outer = firstJsonObject(text)
  if (outer === null) return salvageRecording(text, task)
  const nested = outer['task']
  const raw = nested !== null && typeof nested === 'object' && !Array.isArray(nested)
    ? {
      ...(nested as Record<string, unknown>),
      index_patch: outer['index_patch'] ?? outer['indexPatch'],
      correction_verdicts: outer['correction_verdicts'],
    }
    : outer
  const body = stringField(raw, 'note_body') ?? stringField(raw, 'body') ?? ''
  const summary = coerceWindowSummary(readSummary(raw))
    ?? coerceWindowSummary({ taskDefinition: body, strategyAndProgress: '', keyConclusions: '', deliveryAndAcceptance: '' })
  if (summary === undefined) return salvageRecording(text, task)
  return {
    windowSummary: summary,
    title: stringField(raw, 'title') ?? `turn ${String(task.turn)}`,
    body: body.length > 0 ? body : renderSummaryBody(summary),
    openItems: stringField(raw, 'open_items') ?? '',
    reusableExperience: readExperiences(raw['reusable_experience']),
    topics: readIndexUpdates(raw, 'topics'),
    types: readIndexUpdates(raw, 'types'),
    correctionVerdicts: readVerdicts(raw['correction_verdicts']),
  }
}

/**
 * Read what a truncated reply still contains.
 * @param text - the raw reply.
 * @param task - the span the harness asked for.
 * @returns a recording built from the salvaged fields, or `null` when nothing usable arrived.
 */
function salvageRecording(text: string, task: RecordingTask): Recording | null {
  const fields = salvageFields(text)
  const summary = coerceWindowSummary({
    taskDefinition: fields['taskDefinition'] ?? '',
    strategyAndProgress: fields['strategyAndProgress'] ?? '',
    keyConclusions: fields['keyConclusions'] ?? '',
    deliveryAndAcceptance: fields['deliveryAndAcceptance'] ?? '',
  })
  if (summary === undefined) return null
  const title = fields['title'] ?? ''
  return {
    windowSummary: summary,
    title: title.length > 0 ? title : `turn ${String(task.turn)}`,
    body: renderSummaryBody(summary),
    openItems: fields['openItems'] ?? '',
    reusableExperience: [],
    topics: [],
    types: [],
    correctionVerdicts: [],
  }
}

/**
 * Why one recorder reply could not be read as a patch, in one line.
 * @param text - the recorder's raw reply.
 * @returns the reason, or `null` when the reply actually parses.
 */
export function recordingProblem(text: string): string | null {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text)
  const candidate = fenced?.[1] ?? text
  const start = candidate.indexOf('{')
  const end = candidate.lastIndexOf('}')
  if (start < 0 || end <= start) return `no braces found (reply ${String(text.length)} chars)`
  try {
    JSON.parse(candidate.slice(start, end + 1))
    return null
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    return `invalid JSON: ${message}`
  }
}

/** The first JSON object embedded anywhere in a reply. */
function firstJsonObject(text: string): Record<string, unknown> | null {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text)
  for (const candidate of [fenced?.[1], text]) {
    if (candidate === undefined) continue
    const start = candidate.indexOf('{')
    const end = candidate.lastIndexOf('}')
    if (start < 0 || end <= start) continue
    try {
      const parsed: unknown = JSON.parse(candidate.slice(start, end + 1))
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      continue
    }
  }
  return null
}

/** Read the four window fields, from a nested section or from the object itself. */
function readSummary(raw: Record<string, unknown>): Partial<WindowSummary> {
  const candidate: unknown = raw['window_summary'] ?? raw['windowSummary'] ?? raw
  if (candidate === null || typeof candidate !== 'object' || Array.isArray(candidate)) return {}
  const record = candidate as Record<string, unknown>
  return {
    taskDefinition: pick(record, 'task_definition', 'taskDefinition'),
    strategyAndProgress: pick(record, 'strategy_and_progress', 'strategyAndProgress'),
    keyConclusions: pick(record, 'key_conclusions', 'keyConclusions'),
    deliveryAndAcceptance: pick(record, 'delivery_and_acceptance', 'deliveryAndAcceptance'),
  }
}

/** Read one string under either naming convention. */
function pick(record: Record<string, unknown>, snake: string, camel: string): string {
  return stringField(record, snake) ?? stringField(record, camel) ?? ''
}

/** Read one optional non-empty string field. */
function stringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

/** Render the four fields as the note page. */
function renderSummaryBody(summary: WindowSummary | undefined): string {
  if (summary === undefined) return ''
  return [
    `**Task definition.** ${summary.taskDefinition}`,
    '',
    `**Strategy and progress.** ${summary.strategyAndProgress}`,
    '',
    `**Key conclusions.** ${summary.keyConclusions}`,
    '',
    `**Delivery and acceptance.** ${summary.deliveryAndAcceptance}`,
  ].join('\n')
}

/** Read the reusable-experience list, dropping entries with no lesson text. */
function readExperiences(value: unknown): readonly ReusableExperience[] {
  if (!Array.isArray(value)) return []
  const levels = new Set(['verified', 'partially-verified', 'hypothesis'])
  const entries: ReusableExperience[] = []
  for (const item of value) {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    const experience = pick(record, 'experience', 'lesson')
    if (experience.length === 0) continue
    const level = pick(record, 'verification', 'verification_level')
    entries.push({
      experience,
      verification: (levels.has(level) ? level : 'hypothesis') as VerificationLevel,
      conditions: pick(record, 'conditions', 'when_it_holds'),
      whenToReuse: pick(record, 'when_to_reuse', 'reuse_when'),
      evidence: pick(record, 'evidence', 'source'),
    })
  }
  return entries
}

/** Read the topic/type update lists. */
function readIndexUpdates(
  raw: Record<string, unknown>,
  key: 'topics' | 'types',
): readonly { name: string; description: string }[] {
  const section = raw['index_patch'] ?? raw['indexPatch'] ?? raw['index_updates'] ?? raw['indexUpdates']
  const list = section !== null && typeof section === 'object' && !Array.isArray(section)
    ? (section as Record<string, unknown>)[key]
    : raw[key]
  if (!Array.isArray(list)) return []
  const entries: { name: string; description: string }[] = []
  for (const item of list) {
    if (typeof item === 'string' && item.trim().length > 0) {
      entries.push({ name: item.trim(), description: '' })
      continue
    }
    if (item === null || typeof item !== 'object' || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    const name = pick(record, 'name', 'name')
    if (name.length === 0) continue
    const change = pick(record, 'change', 'change')
    if (change === 'none') continue
    entries.push({ name, description: pick(record, 'description', 'summary') })
  }
  return entries
}

/** Read the correction verdicts. */
function readVerdicts(value: unknown): Recording['correctionVerdicts'] {
  if (!Array.isArray(value)) return []
  const allowed = new Set(['accepted', 'rejected', 'disputed'])
  const entries: { noteId: string; verdict: 'accepted' | 'rejected' | 'disputed'; reason: string }[] = []
  for (const item of value) {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    const noteId = pick(record, 'note_id', 'noteId')
    const verdict = pick(record, 'verdict', 'verdict')
    if (noteId.length === 0 || !allowed.has(verdict)) continue
    entries.push({
      noteId,
      verdict: verdict as 'accepted' | 'rejected' | 'disputed',
      reason: pick(record, 'reason', 'why'),
    })
  }
  return entries
}
