/**
 * The main agent's read-only window tools: the session's notes, a correction
 * channel, and the durable session log.
 *
 * There is deliberately no tool here that writes a note. The agent reads and
 * corrects; the harness records.
 *
 * @module @deepseek-ai/dsh-context-window/tools
 */

import type { Context } from '@deepseek-ai/cordis'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { GenericCallView, ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { ResolvedContextWindowConfig } from './config.ts'
import {
  appendCorrection,
  readIndex,
  readTaskNote,
  renderIndex,
  renderTaskNote,
} from './notes.ts'
import { currentTurnOf } from './window.ts'

/** Model-facing name of the notes tool. */
export const NOTE_TOOL_NAME = 'read_notes'

/** Model-facing name of the correction tool. */
export const NOTE_FIX_TOOL_NAME = 'fix_note'

/** Model-facing name of the tool that reads the durable session log. */
export const LOG_TOOL_NAME = 'session_log'

const NOTE_DESCRIPTION =
  'Read the notes the harness keeps for this session. Call it with op "index" to see every note, its log range, and '
  + 'the topic and type summaries; with op "read" and a note id to read one note in full; with op "search" and a query '
  + 'to find the notes that mention something. Notes are how this session remembers work whose execution trace has '
  + 'left the visible window: read them before assuming, and use the log range each note cites with `session_log` when '
  + 'you need the original events.'

const NOTE_FIX_DESCRIPTION =
  'Report a note that is wrong, stale or misleading, so the note keeper can audit and repair it. State exactly what is '
  + 'wrong, how the note should read, and the evidence (a log seq range, a file path, a command result). The keeper '
  + 'audits every request against the evidence and records its verdict, including a rejection, so your correction will '
  + 'not be silently dropped — but it takes effect from the next recorded turn, not immediately: do not wait for it.'

const LOG_DESCRIPTION =
  'Read the durable session log — the complete record of this session, including the events the sliding window removed '
  + 'from your visible history. Use it when a note cites a log range you need in full: `records: true` lists the spans '
  + 'that were recorded, and a `from`/`to` range (optionally narrowed with `query` or `types`) returns the original '
  + 'events of that range. The log is append-only and complete, so prefer reading it over guessing or redoing work.'

/** Result of one notes read. */
interface NotesValue {
  readonly text: string
  readonly found: boolean
}

/** One rendered log entry returned by `session_log`. */
interface LogEntry {
  readonly seq: number
  readonly type: string
  readonly text: string
}

/** The canonical value of one `session_log` call. */
interface LogValue {
  readonly from: number
  readonly to: number
  readonly matched: number
  readonly returned: number
  readonly truncated: boolean
  readonly entries: LogEntry[]
}

/** Log-only or token-level event types a reader gains nothing from. */
const NOISE_TYPES: ReadonlySet<string> = new Set([
  'assistant/chunk',
  'request/header',
  'request/context',
  'tool/code-dispatch',
  'tool/code-dispatch-start',
  'session/end-seed',
])

/** Default characters returned by one entry before it is elided. */
const ENTRY_MAX_CHARS = 2000

/**
 * Register the read-only window tools into the calling (preset) scope.
 * @param ctx - context the tools register into.
 * @param config - the validated policy of the mounted row.
 */
export function registerWindowTools(ctx: Context, config: ResolvedContextWindowConfig): void {
  ctx.tools.register(defineTool({
    name: NOTE_TOOL_NAME,
    description: NOTE_DESCRIPTION,
    parameters: {
      op: {
        type: 'string',
        enum: ['index', 'read', 'search'],
        description: 'What to return: the note index, one note, or the notes matching a query.',
      },
      note: { type: 'string', description: 'Note id for op "read", e.g. task-0003. Defaults to the newest note.' },
      query: { type: 'string', description: 'Case-insensitive text for op "search".' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: { text: { type: 'string' }, found: { type: 'boolean' } },
      },
      render: (_args, value) => [{ type: 'text', text: String(value.text) }],
    },
    presentCall: (): GenericCallView => ({ card: 'generic', title: 'Read session notes', kind: 'read' }),
    execute(args, exec) {
      const agent = exec.agent
      if (agent === undefined) {
        throw new HarnessError('read_notes needs a live agent session', 'CONTEXT_WINDOW_NO_AGENT')
      }
      return readNotes(config, agent.session, args)
    },
  }))

  ctx.tools.register(defineTool({
    name: NOTE_FIX_TOOL_NAME,
    description: NOTE_FIX_DESCRIPTION,
    parameters: {
      note: { type: 'string', required: true, description: 'The note id that is wrong, e.g. task-0002.' },
      problem: { type: 'string', required: true, description: 'Exactly what is wrong or stale about it.' },
      fix: { type: 'string', required: true, description: 'How the note should read instead.' },
      evidence: { type: 'string', required: true, description: 'The evidence: log seq range, path, or command result.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          accepted_for_audit: { type: 'boolean', required: true },
          note: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: value.accepted_for_audit
          ? `Correction for ${value.note} queued for audit. It takes effect from the next recorded turn; keep working.`
          : `No note ${value.note} exists in this session. Call read_notes (op "index") first.`,
      }],
    },
    presentCall: (args): GenericCallView => ({ card: 'generic', title: `Correct note ${args.note}`, kind: 'other' }),
    async execute(args, exec) {
      const agent = exec.agent
      if (agent === undefined) {
        throw new HarnessError('fix_note needs a live agent session', 'CONTEXT_WINDOW_NO_AGENT')
      }
      const session = agent.session
      const index = await readIndex(config.notesRoot, session)
      if (!index.tasks.some(task => task.id === args.note)) {
        return { accepted_for_audit: false, note: args.note }
      }
      const turn = currentTurnOf(session)
      await appendCorrection(config.notesRoot, session, {
        noteId: args.note,
        problem: args.problem.trim(),
        fix: args.fix.trim(),
        evidence: args.evidence.trim(),
        turn,
        createdAt: new Date().toISOString(),
      })
      ctx.logger.info(`context-window: correction queued for ${args.note} on turn ${String(turn)}`)
      return { accepted_for_audit: true, note: args.note }
    },
  }))

  ctx.tools.register(sessionLogTool(config))
}

/** Read the index, one note, or a search over the stored notes. */
async function readNotes(
  config: ResolvedContextWindowConfig,
  session: Session,
  args: { op?: string | undefined; note?: string | undefined; query?: string | undefined },
): Promise<NotesValue> {
  const index = await readIndex(config.notesRoot, session)
  const op = args.op ?? 'index'
  if (op === 'read') {
    const newest = index.tasks[index.tasks.length - 1]
    const requested = args.note
    const noteId = requested !== undefined && requested.length > 0 ? requested : newest?.id
    if (noteId === undefined) return { text: 'No notes have been recorded for this session yet.', found: false }
    const note = await readTaskNote(config.notesRoot, session, noteId)
    if (note === undefined) {
      return { text: `No note ${noteId} in this session. Call read_notes with op "index" to list them.`, found: false }
    }
    return { text: renderTaskNote(note), found: true }
  }
  if (op === 'search') {
    const query = (args.query ?? '').toLowerCase()
    if (query.length === 0) return { text: 'op "search" needs a query.', found: false }
    const hits: string[] = []
    for (const task of index.tasks) {
      const note = await readTaskNote(config.notesRoot, session, task.id)
      if (note === undefined) continue
      const haystack = `${note.body}\n${note.openItems}\n${JSON.stringify(note.windowSummary)}`.toLowerCase()
      if (haystack.includes(query)) {
        hits.push(`- ${task.id} (turn ${String(task.turn)}, log ${String(task.start)}-${String(task.end)}): ${task.title}`)
      }
    }
    return {
      text: hits.length === 0
        ? `No note mentions "${query}".`
        : `Notes mentioning "${query}":\n${hits.join('\n')}\n\nRead one with read_notes (op "read").`,
      found: hits.length > 0,
    }
  }
  return { text: renderIndex(index), found: index.tasks.length > 0 }
}

/** The read-only log tool. */
function sessionLogTool(config: ResolvedContextWindowConfig): ToolDefinition {
  return defineTool({
    name: LOG_TOOL_NAME,
    description: LOG_DESCRIPTION,
    parameters: {
      from: { type: 'integer', description: 'First log sequence number to include (inclusive). Defaults to the start of the log.' },
      to: { type: 'integer', description: 'Last log sequence number to include (inclusive). Defaults to the end of the log.' },
      query: { type: 'string', description: 'Case-insensitive text filter applied to the returned entries.' },
      types: {
        type: 'array',
        items: { type: 'string' },
        description: 'Restrict raw events to these event types, e.g. ["user/message", "assistant/message"].',
      },
      max_chars: { type: 'integer', description: 'Ceiling on the returned text. Defaults to the configured log read budget.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          from: { type: 'integer', required: true },
          to: { type: 'integer', required: true },
          matched: { type: 'integer', required: true },
          returned: { type: 'integer', required: true },
          truncated: { type: 'boolean', required: true },
          entries: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                seq: { type: 'integer', required: true },
                type: { type: 'string', required: true },
                text: { type: 'string', required: true },
              },
            },
          },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderLog(value) }],
    },
    presentCall: (): GenericCallView => ({ card: 'generic', title: 'Read session log', kind: 'read' }),
    execute(args, exec) {
      const agent = exec.agent
      if (agent === undefined) {
        throw new HarnessError('session_log needs a live agent session', 'CONTEXT_WINDOW_NO_AGENT')
      }
      return Promise.resolve(readLog(agent.session, args, config))
    },
  })
}

/** Read the durable log within the requested bounds. */
function readLog(
  session: Session,
  args: {
    from?: number | undefined
    to?: number | undefined
    query?: string | undefined
    types?: string[] | undefined
    max_chars?: number | undefined
  },
  config: ResolvedContextWindowConfig,
): LogValue {
  const events = session.events
  const last = Math.max(0, events.length - 1)
  const from = clamp(args.from ?? 0, 0, last)
  const to = clamp(args.to ?? last, from, last)
  const budget = args.max_chars === undefined
    ? config.logReadMaxChars
    : clamp(args.max_chars, 1, config.logReadMaxChars)
  const needle = args.query?.toLowerCase()
  const types = args.types === undefined ? undefined : new Set(args.types)

  const entries: LogEntry[] = []
  let matched = 0
  let used = 0
  let truncated = false
  for (let seq = from; seq <= to; seq += 1) {
    const entry = eventEntry(events[seq], seq, types, needle)
    if (entry === undefined) continue
    matched += 1
    const cost = entry.text.length + 32
    if (used + cost > budget) {
      truncated = true
      continue
    }
    used += cost
    entries.push(entry)
  }
  return { from, to, matched, returned: entries.length, truncated, entries }
}

/** Bound one integer argument to an inclusive range. */
function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  const rounded = Math.trunc(value)
  if (rounded < low) return low
  return rounded > high ? high : rounded
}

/** One log event rendered as a log entry, or `undefined` when it is not listed. */
function eventEntry(
  event: SessionEvent | undefined,
  seq: number,
  types: ReadonlySet<string> | undefined,
  needle: string | undefined,
): LogEntry | undefined {
  if (event === undefined) return undefined
  if (types === undefined ? NOISE_TYPES.has(event.type) : !types.has(event.type)) return undefined
  const text = describeEvent(event)
  if (text === undefined || text.length === 0) return undefined
  const body = elide(text)
  if (needle !== undefined && !body.toLowerCase().includes(needle)) return undefined
  return { seq, type: event.type, text: body }
}

/** Render one event's model-relevant content as text. */
function describeEvent(event: SessionEvent): string | undefined {
  switch (event.type) {
    case 'user/message':
      return `USER (${event.data.source.kind}): ${contentText(event.data.content)}`
    case 'assistant/message':
      return `ASSISTANT: ${contentText(event.data.message.content)}`
    case 'tool/result':
      return `RESULT: ${contentText(event.data.message.content)}`
    case 'turn/start':
      return `turn ${String(event.data.turn)} started`
    case 'turn/end':
      return `turn ${String(event.data.turn)} ended (${event.data.reason.kind})`
    case 'step/start':
      return `step ${String(event.data.turn)}.${String(event.data.step)} started`
    case 'step/end':
      return `step ${String(event.data.turn)}.${String(event.data.step)} ended`
    case 'compaction/summary':
      return `compaction summary over log ${String(event.data.shadowedRange.start)}-${String(event.data.shadowedRange.end)}`
    case 'compaction/prune':
      return `window fold over log ${String(event.data.shadowedRange.start)}-${String(event.data.shadowedRange.end)} (~${String(event.data.shadowedTokenCount)} tokens)`
    default:
      return undefined
  }
}

/** Join the text blocks of one content array. */
function contentText(blocks: readonly { type: string; text?: string }[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type !== 'text') continue
    const text = block.text
    if (text !== undefined && text.length > 0) parts.push(text)
  }
  return parts.join('\n').trim()
}

/** Elide one entry that exceeds the per-entry bound. */
function elide(text: string): string {
  return text.length <= ENTRY_MAX_CHARS ? text : `${text.slice(0, ENTRY_MAX_CHARS - 1)}…`
}

/** The model-facing rendering of one log read. */
function renderLog(value: LogValue): string {
  if (value.entries.length === 0) {
    return `No matching log entries between seq ${String(value.from)} and seq ${String(value.to)}.`
  }
  const lines = value.entries.map(entry => `#${String(entry.seq)} [${entry.type}] ${entry.text}`)
  const header = `Session log seq ${String(value.from)}-${String(value.to)}: ${String(value.returned)} of ${String(value.matched)} matching entries${
    value.truncated ? ' (truncated by the read budget; narrow the range)' : ''
  }.`
  return [header, ...lines].join('\n\n')
}
