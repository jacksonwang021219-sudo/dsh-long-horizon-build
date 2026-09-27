/**
 * Durable turn records: what the agent wrote about one user turn, and the log
 * range that record accounts for.
 *
 * A record is a top-level `record_turn` tool result. Its structured payload
 * rides the durable `tool/result.meta` field, so extracting the records of a
 * session is a pure fold over its log: records survive every surface fold, and
 * the window can rebuild its replacement blocks from the log at any time
 * without parsing model-facing prose.
 *
 * @module @deepseek-ai/dsh-context-window/records
 */

import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'

/** Opening tag of a record block the window wrote itself. */
export const RECORD_BLOCK_OPEN = '<turn-record'

/** The `meta.kind` discriminator every record-bearing tool result carries. */
export const RECORD_META_KIND = 'dsh-context-window/record'

/**
 * Why a record covers only part of its turn (`intra-turn`) or the whole turn
 * (`turn`).
 */
export type RecordKind = 'turn' | 'intra-turn'

/** One recorded account of part or all of one user turn. */
export interface ContextRecord {
  /** The user turn this record belongs to. */
  readonly turn: number
  /** Whether the record was written mid-turn under window pressure or at the turn's end. */
  readonly kind: RecordKind
  /** Inclusive first log seq the record accounts for. */
  readonly start: number
  /** Inclusive last log seq the record accounts for. */
  readonly end: number
  /** Priced tokens the accounted range held when the record was written. */
  readonly tokens: number
  /** The recorded account itself. */
  readonly summary: string
  /** Seq of the durable `tool/result` event carrying this record. */
  readonly seq: number
  /** Unix epoch milliseconds when the record was written. */
  readonly time: number
}

/** Incremental record-extraction state for one session. */
interface RecordCache {
  /** Number of log events already scanned. */
  scanned: number
  readonly records: ContextRecord[]
}

const caches = new WeakMap<Session, RecordCache>()

/**
 * Every record the session log holds, in the order the agent wrote them.
 *
 * Extraction is incremental: the log only grows, so each call scans the events
 * appended since the previous one.
 * @param session - session whose log to fold.
 * @returns the session's records, oldest first.
 */
export function sessionRecords(session: Session): readonly ContextRecord[] {
  const events = session.events
  let cache = caches.get(session)
  if (cache === undefined || cache.scanned > events.length) {
    cache = { scanned: 0, records: [] }
    caches.set(session, cache)
  }
  for (let index = cache.scanned; index < events.length; index += 1) {
    const event = events[index]
    const record = recordOfEvent(event) ?? recordOfBlock(event)
    if (record !== undefined) cache.records.push(record)
  }
  cache.scanned = events.length
  return cache.records
}

/**
 * Read one record from a durable tool result, or `undefined` when the event is
 * not a record.
 * @param event - any log event.
 * @returns the record, or `undefined`.
 */
export function recordOfEvent(event: SessionEvent | undefined): ContextRecord | undefined {
  if (event === undefined || event.type !== 'tool/result') return undefined
  const meta = event.data.meta
  if (meta === null || typeof meta !== 'object' || Array.isArray(meta)) return undefined
  const candidate = meta as Record<string, unknown>
  if (candidate['kind'] !== RECORD_META_KIND) return undefined
  const turn = candidate['turn']
  const kind = candidate['recordKind']
  const start = candidate['start']
  const end = candidate['end']
  const tokens = candidate['tokens']
  const summary = candidate['summary']
  if (typeof turn !== 'number' || !Number.isSafeInteger(turn) || turn < 1) return undefined
  if (kind !== 'turn' && kind !== 'intra-turn') return undefined
  if (typeof start !== 'number' || typeof end !== 'number' || !Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return undefined
  if (start < 0 || end < start) return undefined
  if (typeof tokens !== 'number' || !Number.isFinite(tokens) || tokens < 0) return undefined
  if (typeof summary !== 'string' || summary.length === 0) return undefined
  return {
    turn,
    kind,
    start,
    end,
    tokens,
    summary,
    seq: event.seq,
    time: event.time,
  }
}

/**
 * Read one record out of a record BLOCK the window appended, if the event is
 * one.
 *
 * `recordMode: 'recorder'` writes records as window-authored blocks rather than
 * as tool results, and the window is their only producer, so the header it
 * writes is a fixed, parseable form: `<turn-record turn="N" log-range="a-b">`.
 * Parsing it is deterministic — no model-written prose is interpreted.
 * @param event - any log event.
 * @returns the record, or `undefined` when the event is not a record block.
 */
export function recordOfBlock(event: SessionEvent | undefined): ContextRecord | undefined {
  if (event === undefined || event.type !== 'user/message') return undefined
  const source = event.data.source
  if (source.kind !== 'plugin' || source.plugin !== 'dsh-context-window') return undefined
  const header = blockHeader(event.data.content)
  if (header === undefined) return undefined
  return {
    turn: header.turn,
    kind: header.kind,
    start: header.start,
    end: header.end,
    tokens: header.tokens,
    summary: header.summary,
    seq: event.seq,
    time: event.time,
  }
}

/** The fields one record block's header carries, plus its body. */
function blockHeader(blocks: readonly { type: string; text?: string }[]): {
  turn: number
  kind: RecordKind
  start: number
  end: number
  tokens: number
  summary: string
} | undefined {
  const text = blocks
    .map(block => block.type === 'text' ? block.text ?? '' : '')
    .join('\n')
  const open = text.indexOf(RECORD_BLOCK_OPEN)
  if (open !== 0) return undefined
  const close = text.indexOf('>', open)
  if (close < 0) return undefined
  const tag = text.slice(open, close + 1)
  const turn = attribute(tag, 'turn')
  const range = attribute(tag, 'log-range')
  if (turn === undefined || range === undefined) return undefined
  const [rawStart, rawEnd] = range.split('-')
  const start = Number(rawStart)
  const end = rawEnd === undefined ? start : Number(rawEnd)
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return undefined
  const tokens = Number(attribute(tag, 'tokens') ?? 0)
  const body = text.slice(close + 1, text.lastIndexOf('</turn-record>') < 0 ? undefined : text.lastIndexOf('</turn-record>')).trim()
  if (body.length === 0) return undefined
  const kind = attribute(tag, 'kind') === 'intra-turn' ? 'intra-turn' : 'turn'
  return { turn: Number(turn), kind, start, end, tokens: Number.isFinite(tokens) ? tokens : 0, summary: body }
}

/** One quoted attribute of a record block's opening tag. */
function attribute(tag: string, name: string): string | undefined {
  const match = new RegExp(`${name}="([^"]*)"`).exec(tag)
  return match?.[1]
}

/**
 * Whether one log event is a record written by this plugin.
 * @param event - any log event.
 * @returns true for a record-bearing tool result.
 */
export function isRecordEvent(event: SessionEvent | undefined): boolean {
  return recordOfEvent(event) !== undefined
}

/**
 * The records of one turn, oldest first.
 * @param records - every record of the session.
 * @param turn - the turn to filter.
 * @returns that turn's records.
 */
export function recordsForTurn(
  records: readonly ContextRecord[],
  turn: number,
): readonly ContextRecord[] {
  return records.filter(record => record.turn === turn)
}

/**
 * The newest log seq any record of one turn accounts for, or `undefined` when
 * the turn has no records.
 * @param records - every record of the session.
 * @param turn - the turn to measure.
 * @returns the covered-through seq, or `undefined`.
 */
export function coverageEnd(
  records: readonly ContextRecord[],
  turn: number,
): number | undefined {
  let end: number | undefined
  for (const record of records) {
    if (record.turn !== turn) continue
    if (end === undefined || record.end > end) end = record.end
  }
  return end
}

/**
 * The next log seq a record for this turn should start at: immediately after
 * what the turn already accounted for, or `fallback` when nothing has.
 * @param records - every record of the session.
 * @param turn - the turn being recorded.
 * @param fallback - the first seq the turn owns.
 * @returns the inclusive start of the next record's range.
 */
export function nextRecordStart(
  records: readonly ContextRecord[],
  turn: number,
  fallback: number,
): number {
  const end = coverageEnd(records, turn)
  return end === undefined ? fallback : end + 1
}

/** One rendered digest line: what one turn is remembered as. */
export interface DigestEntry {
  /** The turn this line describes. */
  readonly turn: number
  /** Its recorded account, or `undefined` when the turn recorded nothing. */
  readonly summary?: string | undefined
  /** Inclusive log range the turn occupied. */
  readonly start: number
  /** Inclusive log range the turn occupied. */
  readonly end: number
}

/**
 * Render the folded-history digest: one line per turn that left the visible
 * window, newest last, trimmed from the OLDEST end until it fits the budget.
 *
 * Trimming is what keeps a long session's digest bounded; a dropped turn is
 * still fully recoverable, because every record stays in the log and the
 * digest states how many turns it omitted.
 * @param entries - turns to describe, oldest first.
 * @param budgetTokens - ceiling on the rendered text.
 * @param estimate - fixed estimator for that ceiling.
 * @returns the framed digest text.
 */
export function renderHistoryDigest(
  entries: readonly DigestEntry[],
  budgetTokens: number,
  estimate: (text: string) => number,
): string {
  if (entries.length === 0) return ''
  const header = [
    '<folded-history>',
    'Earlier turns left the visible window. Each entry is what you recorded for that turn; the complete original events are still in the session log, so read them with `session_log` (by seq range, or with `records: true` to see every record).',
  ].join('\n')
  const footer = '</folded-history>'
  const lines = entries.map(entry => digestLine(entry))
  const framed = (first: number): string => [
    header,
    ...first === 0 ? [] : [omittedNotice(entries, first)],
    lines.slice(first).join('\n'),
    footer,
  ].join('\n')
  let first = 0
  while (first < entries.length - 1 && estimate(framed(first + 1)) > budgetTokens) first += 1
  return framed(first)
}

/** One digest line: the turn, the log range it occupied, and its account. */
function digestLine(entry: DigestEntry): string {
  const range = entry.start === entry.end
    ? `log ${String(entry.start)}`
    : `log ${String(entry.start)}-${String(entry.end)}`
  if (entry.summary === undefined) {
    return `turn ${String(entry.turn)} (${range}): no account was recorded; read the log range directly if you need it.`
  }
  return `turn ${String(entry.turn)} (${range}):\n${entry.summary}`
}

/** The one-line note listing turns the digest dropped to stay inside its budget. */
function omittedNotice(entries: readonly DigestEntry[], first: number): string {
  const dropped = entries.slice(0, first)
  // oxlint-disable-next-line typescript/no-non-null-assertion -- first > 0 implies at least one dropped entry.
  const oldest = dropped[0]!
  // oxlint-disable-next-line typescript/no-non-null-assertion -- first > 0 implies at least one dropped entry.
  const newest = dropped[dropped.length - 1]!
  return `(${String(dropped.length)} older turn(s), ${String(oldest.turn)}-${String(newest.turn)}, log ${String(oldest.start)}-${String(newest.end)} omitted here; their records remain in the session log.)`
}

/**
 * Render the block that replaces one turn's execution trace: every record the
 * turn wrote, oldest first, each stating the log range it accounts for.
 * @param turn - the turn being replaced.
 * @param records - that turn's records, oldest first.
 * @returns the framed replacement text.
 */
export function renderRecordBlock(turn: number, records: readonly ContextRecord[]): string {
  const header = [
    `<turn-record turn="${String(turn)}" records="${String(records.length)}">`,
    `Your execution trace for turn ${String(turn)} left the visible window. The account below is what you recorded for it; the original events stay in the session log.`,
    'Treat it as established background. When it is not enough, read the cited log range with `session_log` instead of guessing.',
  ].join('\n')
  const body = records.map((record, index) => [
    `[${String(index + 1)}] log seqs ${String(record.start)}-${String(record.end)} (~${String(record.tokens)} tokens, ${record.kind})`,
    record.summary,
  ].join('\n'))
  return [header, ...body, '</turn-record>'].join('\n\n')
}

/**
 * Render the block that replaces a span the window had to fold before the agent
 * recorded it.
 *
 * The in-turn token bound is mandatory — a turn cannot grow without limit just
 * because the agent ignored the window's request — so this block is the
 * pressure valve: it states plainly that no record covers the range and points
 * at the log, where every original event still is.
 * @param turn - the turn being folded.
 * @param start - inclusive first log seq of the folded range.
 * @param end - inclusive last log seq of the folded range.
 * @param tokens - priced tokens the folded span held.
 * @returns the framed replacement text.
 */
export function renderUnrecordedBlock(
  turn: number,
  start: number,
  end: number,
  tokens: number,
): string {
  return [
    `<unrecorded-trace turn="${String(turn)}" log-range="${String(start)}-${String(end)}">`,
    `This turn passed its token limit, so the window folded log seqs ${String(start)}-${String(end)} (~${String(tokens)} tokens) before a record covered them: no summary of it exists yet.`,
    `The original events are untouched in the session log — read them with \`session_log\` (\`from: ${String(start)}\`, \`to: ${String(end)}\`) — and record what still matters with \`record_turn\` when you next need to depend on it.`,
    '</unrecorded-trace>',
  ].join('\n')
}

/**
 * The first log seq of a turn that records have not accounted for; the whole
 * turn when nothing is recorded, or `undefined` when records already cover it.
 * @param records - every record of the session.
 * @param turn - the turn being measured.
 * @param turnStart - the smallest seq the turn owns.
 * @returns the first uncovered seq, or `undefined`.
 */
export function uncoveredStart(
  records: readonly ContextRecord[],
  turn: number,
  turnStart: number,
): number | undefined {
  const end = coverageEnd(records, turn)
  if (end === undefined) return turnStart
  return end + 1
}
