/**
 * The session's note store: durable Markdown-adjacent notes about the work,
 * kept OUTSIDE the agent's workspace and reachable by the agent only through
 * the read tool.
 *
 * One directory per session under the configured root:
 *
 * ```text
 * <root>/<sessionId>/
 *   index.json                  index, topic/type descriptions, task list, corrections
 *   tasks/task-0001.json        one task-level note per recording
 *   topics/<slug>.json          topic-level index + description
 *   types/<slug>.json           type-level index + description
 * ```
 *
 * Notes are stored as canonical JSON (so idempotent writes are a byte compare)
 * and rendered to Markdown on read (so the agent gets prose, not JSON). Every
 * note carries the log range it covers, which is the only handle the agent has
 * on content the sliding window has replaced.
 *
 * @module @deepseek-ai/dsh-context-window/notes
 */

import { appendFile, mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { Session } from '@deepseek-ai/dsh-session'

/** Which level of the note hierarchy one note belongs to. */
export type NoteKind = 'task' | 'topic' | 'type'

/** How well verified one reusable experience is. */
export type VerificationLevel = 'verified' | 'partially-verified' | 'hypothesis'

/** One reusable lesson, with the conditions that make it reusable. */
export interface ReusableExperience {
  /** The lesson itself. */
  readonly experience: string
  /** How far it has been verified in this session. */
  readonly verification: VerificationLevel
  /** The preconditions under which it holds. */
  readonly conditions: string
  /** When reusing it is worth considering. */
  readonly whenToReuse: string
  /** The log range or workspace facts it rests on. */
  readonly evidence: string
}

/** The four fields the sliding window keeps for one turn. */
export interface WindowSummary {
  readonly taskDefinition: string
  readonly strategyAndProgress: string
  readonly keyConclusions: string
  readonly deliveryAndAcceptance: string
}

/**
 * One task-level note.
 *
 * The four `windowSummary` fields ARE the note: a note is a short page of
 * conclusions a later turn can read at once, not a transcript of the work. The
 * `body` is their rendering and exists so a reader gets one coherent page.
 */
export interface TaskNote {
  /** Stable note id, `task-0001`. */
  readonly id: string
  /** The turn this note belongs to. */
  readonly turn: number
  /** Inclusive first log seq the note accounts for. */
  readonly start: number
  /** Inclusive last log seq the note accounts for. */
  readonly end: number
  /** When the note was written (ISO 8601). */
  readonly createdAt: string
  /** The four window fields, rendered into the visible surface. */
  readonly windowSummary: WindowSummary
  /** The full task note body. */
  readonly body: string
  /** Open items and the next actionable step. */
  readonly openItems: string
  /** Lessons a later session could reuse, if any. */
  readonly reusableExperience: readonly ReusableExperience[]
}

/** One topic- or type-level description, with its index of task notes. */
export interface IndexNote {
  /** Stable slug. */
  readonly slug: string
  /** Human-readable name. */
  readonly name: string
  /** The description the recorder maintains. */
  readonly description: string
  /** Task-note ids this entry indexes, oldest first. */
  readonly taskIds: readonly string[]
  /** When the description last changed (ISO 8601). */
  readonly updatedAt: string
}

/** A correction the main agent asked for, and the recorder's audit of it. */
export interface CorrectionRequest {
  /** The note the agent believes is wrong. */
  readonly noteId: string
  /** What is wrong, specifically. */
  readonly problem: string
  /** How the agent thinks it should read instead. */
  readonly fix: string
  /** The evidence the agent offers (log range, path, command output). */
  readonly evidence: string
  /** Turn that raised it. */
  readonly turn: number
  /** When it was raised (ISO 8601). */
  readonly createdAt: string
  /** The recorder's verdict, once it has audited the request. */
  readonly verdict?: 'accepted' | 'rejected' | 'disputed'
  /** Why the recorder decided that. */
  readonly verdictReason?: string
}

/** The index file one session keeps. */
export interface NoteIndex {
  readonly sessionId: string
  readonly createdAt: string
  /** Task notes, oldest first, with their log ranges. */
  readonly tasks: readonly IndexTask[]
  /** Topic-level entries. */
  readonly topics: readonly IndexNote[]
  /** Type-level entries. */
  readonly types: readonly IndexNote[]
  /** Corrections raised by the main agent and the recorder's verdicts. */
  readonly corrections: readonly CorrectionRequest[]
}

/**
 * One task note as the index carries it.
 *
 * `conclusions` is the note's `keyConclusions` field, capped at
 * {@link CONCLUSION_DIGEST_CHARS}. The index is what every later recording and
 * every folded-history digest reads, so it must state what a note already
 * concluded: a bare title leaves the dedup rule with nothing to apply to.
 */
export interface IndexTask {
  /** Stable note id, `task-0001`. */
  readonly id: string
  /** The turn this note belongs to. */
  readonly turn: number
  /** Inclusive first log seq the note accounts for. */
  readonly start: number
  /** Inclusive last log seq the note accounts for. */
  readonly end: number
  /** Short title for the note. */
  readonly title: string
  /** The note's key conclusions, truncated to the index's own budget. */
  readonly conclusions: string
}

/** Characters of a note's conclusions an index entry keeps. */
export const CONCLUSION_DIGEST_CHARS = 400

/** What one recording produced, before it is written. */
export interface NoteDraft {
  /** The turn recorded. */
  readonly turn: number
  /** Inclusive first log seq covered. */
  readonly start: number
  /** Inclusive last log seq covered. */
  readonly end: number
  /** Short title for index entries. */
  readonly title: string
  /** The four window fields. */
  readonly windowSummary: WindowSummary
  /** The full note body. */
  readonly body: string
  /** Open items and next step. */
  readonly openItems: string
  /** Reusable lessons, possibly empty. */
  readonly reusableExperience: readonly ReusableExperience[]
  /** Topic entries the recorder wants created or updated. */
  readonly topics: readonly { name: string; description: string }[]
  /** Type entries the recorder wants created or updated. */
  readonly types: readonly { name: string; description: string }[]
  /** Audits of corrections the main agent raised. */
  readonly correctionVerdicts: readonly {
    noteId: string
    verdict: 'accepted' | 'rejected' | 'disputed'
    reason: string
  }[]
}

/**
 * Append one diagnostic line to the session's recorder log.
 *
 * The harness logger is not reachable from every deployment surface, and a
 * recording that fails silently is the one failure mode this policy cannot
 * tolerate, so failures are written beside the notes where an operator (or the
 * next run) can read them.
 * @param root - the notes root directory.
 * @param session - the session the recording belonged to.
 * @param line - the diagnostic line.
 */
export async function appendRecorderLog(root: string, session: Session, line: string): Promise<void> {
  const dir = sessionNotesDir(root, session)
  try {
    await mkdir(dir, { recursive: true })
    await appendFile(join(dir, 'recorder.log'), `${new Date().toISOString()} ${line}\n`, 'utf8')
  } catch {
    // A diagnostic that cannot be written must not break the turn.
  }
}

/** Where one session's notes live. */
export function sessionNotesDir(root: string, session: Session): string {
  return join(root, session.id)
}

/** One recorder call's provider accounting, as persisted for cost reporting. */
export interface RecorderUsageEntry {
  /** The turn the call recorded. */
  readonly turn: number
  /** Which attempt succeeded or, on a failure, which one it was. */
  readonly attempt: number
  /** Whether the reply was accepted as a recording. */
  readonly accepted: boolean
  /** Provider route the call was billed to. */
  readonly provider: string
  /** Model the call was billed to. */
  readonly model: string
  /** Wall-clock milliseconds the call took. */
  readonly durationMs: number
  /** Input tokens the provider did not serve from its prefix cache. */
  readonly inputTokens: number
  /** Generated tokens, reasoning included. */
  readonly outputTokens: number
  /** Input tokens the provider served from its prefix cache. */
  readonly cacheReadTokens: number
  /** Input tokens the provider charged as a cache write. */
  readonly cacheWriteTokens: number
  /** Reasoning tokens, when the provider reports them separately. */
  readonly reasoningTokens: number
}

/**
 * Append one recorder call's accounting to the session's usage log.
 *
 * The recorder's own model calls never enter the session log, so this file is
 * the only place the cost of a note can be priced from. One JSON object per
 * call keeps it append-only, and a call that cannot be written is dropped
 * rather than allowed to break the turn it was recording.
 * @param root - the notes root directory.
 * @param session - the session the recording belonged to.
 * @param entry - the call's accounting.
 */
export async function appendRecorderUsage(
  root: string,
  session: Session,
  entry: RecorderUsageEntry,
): Promise<void> {
  const dir = sessionNotesDir(root, session)
  try {
    await mkdir(dir, { recursive: true })
    await appendFile(join(dir, 'recorder-usage.jsonl'), `${JSON.stringify(entry)}\n`, 'utf8')
  } catch {
    // Cost accounting that cannot be written must not break the turn.
  }
}

/** Canonical JSON: sorted keys, two-space indent, trailing newline. */
function canonical(value: unknown): string {
  const walk = (input: unknown): unknown => {
    if (Array.isArray(input)) return input.map(entry => walk(entry))
    if (input === null || typeof input !== 'object') return input
    const entries = Object.entries(input as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    return Object.fromEntries(entries.map(([key, entry]) => [key, walk(entry)]))
  }
  return `${JSON.stringify(walk(value), null, 2)}\n`
}

/** Write one file only when its canonical content changed. */
async function writeIfChanged(path: string, content: string): Promise<boolean> {
  try {
    const existing = await readFile(path, 'utf8')
    if (existing === content) return false
  } catch {
    // Absent file: fall through and write it.
  }
  const temporary = `${path}.tmp`
  await writeFile(temporary, content, 'utf8')
  await rename(temporary, path)
  return true
}

/** Turn a free-form name into a stable file slug. */
export function slugify(name: string): string {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return slug.length === 0 ? 'untitled' : slug.slice(0, 64)
}

/** Read the index of one session, or an empty one when nothing is stored yet. */
export async function readIndex(root: string, session: Session): Promise<NoteIndex> {
  const path = join(sessionNotesDir(root, session), 'index.json')
  try {
    return normalizeIndex(JSON.parse(await readFile(path, 'utf8')) as NoteIndex)
  } catch {
    return {
      sessionId: session.id,
      createdAt: new Date().toISOString(),
      tasks: [],
      topics: [],
      types: [],
      corrections: [],
    }
  }
}

/**
 * Fill in index fields an earlier writer did not record.
 *
 * `index.json` is a durable file, so a field added later is absent in notes
 * written before it existed.
 */
function normalizeIndex(index: NoteIndex): NoteIndex {
  return {
    ...index,
    tasks: index.tasks.map(task => ({
      ...task,
      conclusions: typeof task.conclusions === 'string' ? task.conclusions : '',
    })),
  }
}

/** Cap one note's conclusions at what an index entry carries, on one line. */
function digestConclusions(text: string): string {
  const flattened = text.trim().replace(/\s+/g, ' ')
  if (flattened.length <= CONCLUSION_DIGEST_CHARS) return flattened
  return `${flattened.slice(0, CONCLUSION_DIGEST_CHARS - 1).trimEnd()}…`
}

/**
 * Write one recording into the session's note store.
 *
 * A task note is always appended (each recording is its own evidence); topic,
 * type, and correction entries are only rewritten when their content actually
 * changes, so an unchanged index produces no file diff.
 * @param root - the notes root directory.
 * @param session - the session the notes belong to.
 * @param draft - the recording to store.
 * @returns what happened, for the caller's log line.
 */
export async function writeRecording(
  root: string,
  session: Session,
  draft: NoteDraft,
): Promise<{ noteId: string; wroteTask: boolean; changedIndex: boolean }> {
  const dir = sessionNotesDir(root, session)
  await mkdir(join(dir, 'tasks'), { recursive: true })
  await mkdir(join(dir, 'topics'), { recursive: true })
  await mkdir(join(dir, 'types'), { recursive: true })

  const index = await readIndex(root, session)
  const noteId = `task-${String(index.tasks.length + 1).padStart(4, '0')}`
  const note: TaskNote = {
    id: noteId,
    turn: draft.turn,
    start: draft.start,
    end: draft.end,
    createdAt: new Date().toISOString(),
    windowSummary: draft.windowSummary,
    body: draft.body,
    openItems: draft.openItems,
    reusableExperience: draft.reusableExperience,
  }
  const wroteTask = await writeIfChanged(join(dir, 'tasks', `${noteId}.json`), canonical(note))

  const next: NoteIndex = {
    ...index,
    tasks: [...index.tasks, {
      id: noteId,
      turn: draft.turn,
      start: draft.start,
      end: draft.end,
      title: draft.title,
      conclusions: digestConclusions(draft.windowSummary.keyConclusions),
    }],
    topics: mergeIndexNotes(index.topics, draft.topics, noteId),
    types: mergeIndexNotes(index.types, draft.types, noteId),
    corrections: applyVerdicts(index.corrections, draft.correctionVerdicts),
  }
  for (const entry of [...next.topics]) {
    await writeIfChanged(join(dir, 'topics', `${entry.slug}.json`), canonical(entry))
  }
  for (const entry of [...next.types]) {
    await writeIfChanged(join(dir, 'types', `${entry.slug}.json`), canonical(entry))
  }
  const changedIndex = await writeIfChanged(join(dir, 'index.json'), canonical(next))
  return { noteId, wroteTask, changedIndex }
}

/**
 * Merge the recorder's topic/type updates into the index.
 *
 * An entry is created when it is new, its description replaced when the
 * recorder supplied a different one, and left alone otherwise — the index does
 * not churn, and nothing is ever deleted.
 */
function mergeIndexNotes(
  existing: readonly IndexNote[],
  updates: readonly { name: string; description: string }[],
  noteId: string,
): readonly IndexNote[] {
  const merged = existing.map(entry => ({ ...entry, taskIds: [...entry.taskIds] }))
  for (const update of updates) {
    const slug = slugify(update.name)
    const found = merged.find(entry => entry.slug === slug)
    if (found === undefined) {
      merged.push({
        slug,
        name: update.name,
        description: update.description,
        taskIds: [noteId],
        updatedAt: new Date().toISOString(),
      })
      continue
    }
    found.taskIds.push(noteId)
    const description = update.description.trim()
    if (description.length > 0 && description !== found.description) {
      merged[found.slug === slug ? merged.indexOf(found) : 0] = {
        ...found,
        description,
        updatedAt: new Date().toISOString(),
      }
    }
  }
  return merged
}

/** Record the recorder's verdicts on the main agent's corrections. */
function applyVerdicts(
  existing: readonly CorrectionRequest[],
  verdicts: readonly { noteId: string; verdict: 'accepted' | 'rejected' | 'disputed'; reason: string }[],
): readonly CorrectionRequest[] {
  if (verdicts.length === 0) return existing
  return existing.map((entry) => {
    if (entry.verdict !== undefined) return entry
    const verdict = verdicts.find(candidate => candidate.noteId === entry.noteId)
    if (verdict === undefined) return entry
    return { ...entry, verdict: verdict.verdict, verdictReason: verdict.reason }
  })
}

/** Append one correction request and return the updated index. */
export async function appendCorrection(
  root: string,
  session: Session,
  correction: CorrectionRequest,
): Promise<NoteIndex> {
  const dir = sessionNotesDir(root, session)
  await mkdir(dir, { recursive: true })
  const index = await readIndex(root, session)
  const next: NoteIndex = { ...index, corrections: [...index.corrections, correction] }
  await writeIfChanged(join(dir, 'index.json'), canonical(next))
  return next
}

/** Read one stored task note, or `undefined` when the id is unknown. */
export async function readTaskNote(
  root: string,
  session: Session,
  noteId: string,
): Promise<TaskNote | undefined> {
  try {
    return JSON.parse(await readFile(join(sessionNotesDir(root, session), 'tasks', `${noteId}.json`), 'utf8')) as TaskNote
  } catch {
    return undefined
  }
}

/** Render one task note as the Markdown the agent reads. */
export function renderTaskNote(note: TaskNote): string {
  const lines = [
    `# Task note ${note.id} (turn ${String(note.turn)})`,
    '',
    `Log range: seq ${String(note.start)}-${String(note.end)} — read it back with \`session_log\` (from=${String(note.start)}, to=${String(note.end)}).`,
    '',
    note.body.trim(),
  ]
  if (note.openItems.trim().length > 0) lines.push('', '## Open items and next step', '', note.openItems.trim())
  if (note.reusableExperience.length > 0) {
    lines.push('', '## Reusable experience')
    for (const entry of note.reusableExperience) {
      lines.push(
        '',
        `- **${entry.experience}**`,
        `  - verification: ${entry.verification}`,
        `  - holds when: ${entry.conditions}`,
        `  - worth reusing when: ${entry.whenToReuse}`,
        `  - evidence: ${entry.evidence}`,
      )
    }
  }
  return lines.join('\n')
}

/**
 * Render everything the recorder needs to know about the notes that already
 * exist: every topic and type entry in full, and every task note's title and
 * key conclusions.
 *
 * It is handed over complete rather than summarized, because the recorder's
 * only job is to decide whether this turn changes any of it — and that decision
 * needs the current text, not a paraphrase of it.
 * @param index - the session's note index.
 * @returns the context block for the recorder's prompt.
 */
export function renderNotesContext(index: NoteIndex): string {
  const lines = [
    '<existing-notes>',
    `Task notes so far (${String(index.tasks.length)}):`,
    ...index.tasks.length === 0
      ? ['(none yet)']
      : index.tasks.flatMap(task => [
        `- ${task.id} (turn ${String(task.turn)}, log ${String(task.start)}-${String(task.end)}): ${task.title}`,
        ...task.conclusions.length === 0 ? [] : [`  conclusions already recorded: ${task.conclusions}`],
      ]),
    '',
    `Topics (${String(index.topics.length)}):`,
    ...index.topics.length === 0
      ? ['(none yet)']
      : index.topics.map(topic => `- "${topic.name}" [tasks: ${topic.taskIds.join(', ') || 'none'}]: ${topic.description || '(no description)'}`),
    '',
    `Types (${String(index.types.length)}):`,
    ...index.types.length === 0
      ? ['(none yet)']
      : index.types.map(type => `- "${type.name}" [tasks: ${type.taskIds.join(', ') || 'none'}]: ${type.description || '(no description)'}`),
    ...index.corrections.filter(entry => entry.verdict === undefined).length === 0 ? [] : [
      '',
      'Corrections awaiting your audit:',
      ...index.corrections.filter(entry => entry.verdict === undefined).map(entry =>
        `- ${entry.noteId}: "${entry.problem}" (suggested reading: "${entry.fix}"; evidence: ${entry.evidence})`),
    ],
    '</existing-notes>',
  ]
  return lines.join('\n')
}

/** Render the index as the Markdown the agent reads. */
export function renderIndex(index: NoteIndex): string {
  const lines = [
    `# Notes for this session (${String(index.tasks.length)} task note(s))`,
    '',
    'Every note states the log seq range it covers; read the original events with `session_log`.',
  ]
  if (index.topics.length > 0) {
    lines.push('', '## Topics')
    for (const topic of index.topics) {
      lines.push(`- **${topic.name}** (${topic.taskIds.join(', ')}): ${topic.description || '(no description yet)'}`)
    }
  }
  if (index.types.length > 0) {
    lines.push('', '## Types')
    for (const type of index.types) {
      lines.push(`- **${type.name}** (${type.taskIds.join(', ')}): ${type.description || '(no description yet)'}`)
    }
  }
  if (index.tasks.length > 0) {
    lines.push('', '## Task notes')
    for (const task of index.tasks) {
      lines.push(`- ${task.id} — turn ${String(task.turn)}, log ${String(task.start)}-${String(task.end)}: ${task.title}`)
      if (task.conclusions.length > 0) lines.push(`  - conclusions: ${task.conclusions}`)
    }
  }
  const open = index.corrections.filter(entry => entry.verdict === undefined)
  if (open.length > 0) {
    lines.push('', '## Corrections awaiting audit')
    for (const entry of open) lines.push(`- ${entry.noteId}: ${entry.problem}`)
  }
  return lines.join('\n')
}

/**
 * The fixed line every folded record opens with.
 *
 * A fold puts a record where the request-and-answer pair it replaced used to
 * sit, so the record is read as the answer this session gave to that request.
 * The line states that in the same words every time, which is what lets the
 * main agent recognize a folded record as one of its own past answers rather
 * than a document it is being handed.
 */
export const FOLDED_RECORD_PREFIX = '针对本问题Agent的任务执行的大致情况如下：'

/** Render the window summary one folded turn is replaced by. */
export function renderWindowSummary(turn: number, start: number, end: number, summary: WindowSummary, noteId: string): string {
  return [
    `<turn-summary turn="${String(turn)}" note="${noteId}" log-range="${String(start)}-${String(end)}">`,
    `Your execution trace for turn ${String(turn)} left the visible window. What follows is the record kept for it; the original events stay in the session log (seq ${String(start)}-${String(end)}).`,
    '',
    FOLDED_RECORD_PREFIX,
    '',
    `**Task definition.** ${summary.taskDefinition}`,
    '',
    `**Strategy and progress.** ${summary.strategyAndProgress}`,
    '',
    `**Key conclusions.** ${summary.keyConclusions}`,
    '',
    `**Delivery and acceptance.** ${summary.deliveryAndAcceptance}`,
    '',
    `Read the full note with the notes tool (${noteId}) when you need more, or the original events with \`session_log\`.`,
    '</turn-summary>',
  ].join('\n')
}

/**
 * Salvage the fields of a truncated patch reply.
 *
 * A reply that ran out of generation budget is still most of an answer: the
 * window summary sits at the top of the contract, so the fields that did arrive
 * can be recovered with a scan. This keeps a turn's record usable instead of
 * losing it to one missing closing brace.
 * @param text - the raw reply.
 * @returns the salvaged strings, each empty when its field never arrived.
 */
export function salvageFields(text: string): Record<string, string> {
  const grab = (key: string): string => {
    const match = new RegExp(`"${key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"?`).exec(text)
    if (match?.[1] === undefined) return ''
    try {
      return JSON.parse(`"${match[1]}"`) as string
    } catch {
      return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').trim()
    }
  }
  return {
    title: grab('title'),
    taskDefinition: grab('task_definition'),
    strategyAndProgress: grab('strategy_and_progress'),
    keyConclusions: grab('key_conclusions'),
    deliveryAndAcceptance: grab('delivery_and_acceptance'),
    noteBody: grab('note_body'),
    openItems: grab('open_items'),
  }
}

/** The four window fields, extracted from a looser recording shape. */
export function coerceWindowSummary(value: Partial<WindowSummary> | undefined): WindowSummary | undefined {
  if (value === undefined) return undefined
  const fields: WindowSummary = {
    taskDefinition: (value.taskDefinition ?? '').trim(),
    strategyAndProgress: (value.strategyAndProgress ?? '').trim(),
    keyConclusions: (value.keyConclusions ?? '').trim(),
    deliveryAndAcceptance: (value.deliveryAndAcceptance ?? '').trim(),
  }
  const values: string[] = [fields.taskDefinition, fields.strategyAndProgress, fields.keyConclusions, fields.deliveryAndAcceptance]
  return values.some(entry => entry.length > 0) ? fields : undefined
}
