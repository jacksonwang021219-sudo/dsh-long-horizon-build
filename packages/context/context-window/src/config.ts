/**
 * Load-time validation for the sliding-window context policy.
 *
 * @module @deepseek-ai/dsh-context-window/config
 */

import { deepFreeze } from '@deepseek-ai/dsh-llm'

/**
 * Sliding-window retention, in-turn folding, and record policy.
 *
 * The window is defined over USER REQUESTS, not over request-and-answer pairs:
 * the most recent {@link ContextWindowConfig.retainUserTurns} prompts stay on
 * the model-visible surface, while the execution traces that answered them are
 * replaced by the records the agent itself wrote. `retainTurnTokens` and
 * `foldTriggerTokens` bound how much of one still-running turn may stay
 * before the same rule applies inside the turn.
 */
export interface ContextWindowConfig {
  /**
   * User requests kept verbatim on the model-visible surface, counting the
   * request being prepared. Defaults to `5`. Earlier requests leave the
   * surface; their records stay in the folded-history digest.
   */
  retainUserTurns?: number
  /**
   * Trajectory tokens kept for the turn in progress once an in-turn fold has
   * run, counted from the newest node backwards. Defaults to `200000`.
   */
  retainTurnTokens?: number
  /**
   * Trajectory tokens within one user turn that trigger an in-turn fold.
   * Defaults to `400000`, and must exceed `retainTurnTokens` so a fold always
   * returns the turn below its own trigger. The trigger is rounded up to a
   * whole agent step, so a fold never cuts a tool call from its result.
   */
  foldTriggerTokens?: number
  /** Ceiling on the rendered folded-history digest, in estimated tokens. Defaults to `8192`. */
  digestMaxTokens?: number
  /** Ceiling on one recorded summary, in characters. Defaults to `8000`. */
  recordMaxChars?: number
  /** Ceiling on one `session_log` result, in characters. Defaults to `60000`. */
  logReadMaxChars?: number
  /**
   * Record reminders the end-of-turn gate may steer per turn before it gives
   * up and lets the turn close unrecorded. Defaults to `2`.
   */
  recordReminders?: number
  /**
   * Who writes a turn's record.
   *
   * `recorder` is the only mode: the window writes the record itself, with one
   * model call over the folded span, at the moment a fold needs it. The main
   * agent is never asked to record and there is no end-of-turn gate, so a
   * session whose window never folds pays nothing. The value stays explicit so
   * that an obsolete composition fails at load instead of being ignored.
   */
  recordMode?: 'recorder'
  /**
   * How much the system prompt says about records.
   *
   * - `full` (default): the standing obligation to record every turn, plus the
   *   content checklist and the read-back rules.
   * - `minimal`: read-only guidance, which is all the main agent needs while
   *   the recorder writes every note.
   */
  policy?: 'full' | 'minimal'
  /** Generation ceiling for one recorder call, in tokens. Defaults to `2048`. */
  recorderMaxTokens?: number
  /**
   * The message history one recorder call replays.
   *
   * - `aligned` (default): the model-visible window the main agent itself
   *   sends — the retained requests, the summaries that replaced earlier
   *   traces, and this turn's trace. The request then shares the conversation's
   *   own prefix, so the provider bills it at the cache-hit price and the note
   *   keeper judges this turn against what the main agent can still see.
   * - `span`: only the recorded turn's own messages. The request diverges from
   *   the conversation at its first message, so the provider bills the whole
   *   prompt at the cache-miss price, and the note keeper cannot see which
   *   conclusions an earlier summary already carries.
   */
  recorderView?: 'aligned' | 'span'
  /**
   * Whether the recording contract forbids repeating conclusions that are
   * already covered by the notes or visible in the replayed window. Defaults to
   * `true`; it is only enforceable in the `aligned` view, where the earlier
   * summaries are part of the request.
   */
  recorderDedup?: boolean
  /**
   * Root directory for the session's notes, outside any workspace. Defaults to
   * `<harness home>/notes`. One directory per session is created under it.
   */
  notesRoot?: string
  /** Recording attempts before the turn degrades. Defaults to `5`. */
  recorderRetries?: number
  /** Wall-clock ceiling for one recording attempt, in milliseconds. Defaults to `600000`. */
  recorderTimeoutMs?: number
  /**
   * Milliseconds the next turn waits for an unfinished recording before giving
   * up and sending the real request with the window unfolded. Defaults to `120000`.
   */
  recorderWaitMs?: number
  /**
   * Consecutive turns allowed to degrade before the whole session stops folding
   * and recording. Defaults to `3`.
   */
  recorderCircuitBreaker?: number
}

/** The validated, frozen policy one mounted row runs. */
export interface ResolvedContextWindowConfig {
  readonly retainUserTurns: number
  readonly retainTurnTokens: number
  readonly foldTriggerTokens: number
  readonly digestMaxTokens: number
  readonly recordMaxChars: number
  readonly logReadMaxChars: number
  readonly recordReminders: number
  readonly recordMode: 'recorder'
  readonly policy: 'full' | 'minimal'
  readonly recorderMaxTokens: number
  readonly recorderView: 'aligned' | 'span'
  readonly recorderDedup: boolean
  readonly notesRoot: string
  readonly recorderRetries: number
  readonly recorderTimeoutMs: number
  readonly recorderWaitMs: number
  readonly recorderCircuitBreaker: number
}

/**
 * The default notes root: the harness home's `notes/` directory, deliberately
 * outside every workspace so a task's own file operations cannot touch it.
 */
function defaultNotesRoot(): string {
  const home = process.env['DSH_HOME'] ?? `${process.env['HOME'] ?? '.'}/.dsh`
  return `${home}/notes`
}

/** Complete public configuration key set; anything else is a typo. */
const CONFIG_KEYS: ReadonlySet<string> = new Set([
  'retainUserTurns',
  'retainTurnTokens',
  'foldTriggerTokens',
  'digestMaxTokens',
  'recordMaxChars',
  'logReadMaxChars',
  'recordReminders',
  'recordMode',
  'policy',
  'recorderMaxTokens',
  'recorderView',
  'recorderDedup',
  'notesRoot',
  'recorderRetries',
  'recorderTimeoutMs',
  'recorderWaitMs',
  'recorderCircuitBreaker',
])

/** User requests retained verbatim when the row sets nothing. */
export const DEFAULT_RETAIN_USER_TURNS = 5

/** Trajectory tokens one turn keeps after an in-turn fold when the row sets nothing. */
export const DEFAULT_RETAIN_TURN_TOKENS = 200_000

/** In-turn trajectory tokens that trigger a fold when the row sets nothing. */
export const DEFAULT_FOLD_TRIGGER_TOKENS = 400_000

/**
 * Validate and freeze one row's configuration.
 * @param config - untrusted plugin configuration after Loader normalization.
 * @returns detached immutable policy.
 * @throws when a key is unknown or a value is outside its documented domain.
 */
export function resolveContextWindowConfig(
  config: ContextWindowConfig = {},
): ResolvedContextWindowConfig {
  for (const key of Object.keys(config)) {
    if (!CONFIG_KEYS.has(key)) throw new Error(`ContextWindowConfig: unknown key "${key}"`)
  }
  const retainUserTurns = config.retainUserTurns ?? DEFAULT_RETAIN_USER_TURNS
  assertPositiveInteger('ContextWindowConfig.retainUserTurns', retainUserTurns)
  const retainTurnTokens = config.retainTurnTokens ?? DEFAULT_RETAIN_TURN_TOKENS
  assertPositiveInteger('ContextWindowConfig.retainTurnTokens', retainTurnTokens)
  const foldTriggerTokens = config.foldTriggerTokens ?? DEFAULT_FOLD_TRIGGER_TOKENS
  assertPositiveInteger('ContextWindowConfig.foldTriggerTokens', foldTriggerTokens)
  if (foldTriggerTokens <= retainTurnTokens) {
    throw new Error(
      `ContextWindowConfig.foldTriggerTokens (${String(foldTriggerTokens)}) must exceed `
      + `ContextWindowConfig.retainTurnTokens (${String(retainTurnTokens)}), otherwise a fold `
      + 'returns the turn to its own trigger and every later step folds again',
    )
  }
  const digestMaxTokens = config.digestMaxTokens ?? 8192
  assertPositiveInteger('ContextWindowConfig.digestMaxTokens', digestMaxTokens)
  const recordMaxChars = config.recordMaxChars ?? 8000
  assertPositiveInteger('ContextWindowConfig.recordMaxChars', recordMaxChars)
  const logReadMaxChars = config.logReadMaxChars ?? 60_000
  assertPositiveInteger('ContextWindowConfig.logReadMaxChars', logReadMaxChars)
  const recordReminders = config.recordReminders ?? 2
  assertNonNegativeInteger('ContextWindowConfig.recordReminders', recordReminders)
  const recordMode = readEnum('ContextWindowConfig.recordMode', config.recordMode, ['recorder'], 'recorder')
  const policy = readEnum('ContextWindowConfig.policy', config.policy, ['full', 'minimal'], 'full')
  const recorderMaxTokens = config.recorderMaxTokens ?? 2048
  assertPositiveInteger('ContextWindowConfig.recorderMaxTokens', recorderMaxTokens)
  const recorderView = readEnum('ContextWindowConfig.recorderView', config.recorderView, ['aligned', 'span'], 'aligned')
  const recorderDedup = config.recorderDedup ?? true
  if (typeof recorderDedup !== 'boolean') {
    throw new Error(`ContextWindowConfig.recorderDedup (${String(recorderDedup)}) must be a boolean`)
  }
  const notesRoot = config.notesRoot ?? defaultNotesRoot()
  if (typeof notesRoot !== 'string' || notesRoot.length === 0) {
    throw new Error('ContextWindowConfig.notesRoot must be a non-empty path')
  }
  const recorderRetries = config.recorderRetries ?? 5
  assertPositiveInteger('ContextWindowConfig.recorderRetries', recorderRetries)
  const recorderTimeoutMs = config.recorderTimeoutMs ?? 600_000
  assertPositiveInteger('ContextWindowConfig.recorderTimeoutMs', recorderTimeoutMs)
  const recorderWaitMs = config.recorderWaitMs ?? 120_000
  assertPositiveInteger('ContextWindowConfig.recorderWaitMs', recorderWaitMs)
  const recorderCircuitBreaker = config.recorderCircuitBreaker ?? 3
  assertPositiveInteger('ContextWindowConfig.recorderCircuitBreaker', recorderCircuitBreaker)
  return deepFreeze({
    retainUserTurns,
    retainTurnTokens,
    foldTriggerTokens,
    digestMaxTokens,
    recordMaxChars,
    logReadMaxChars,
    recordReminders,
    recordMode,
    policy,
    recorderMaxTokens,
    recorderView,
    recorderDedup,
    notesRoot,
    recorderRetries,
    recorderTimeoutMs,
    recorderWaitMs,
    recorderCircuitBreaker,
  })
}

/** Validate one enumerated string field, defaulting when the row omits it. */
function readEnum<T extends string>(
  name: string,
  value: unknown,
  allowed: readonly T[],
  fallback: T,
): T {
  if (value === undefined) return fallback
  const match = allowed.find(candidate => candidate === value)
  if (match === undefined) {
    throw new Error(`${name} must be one of ${allowed.map(entry => JSON.stringify(entry)).join(', ')}, received ${JSON.stringify(value)}`)
  }
  return match
}

function assertPositiveInteger(name: string, value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} (${String(value)}) must be a positive integer`)
  }
}

function assertNonNegativeInteger(name: string, value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new Error(`${name} (${String(value)}) must be a non-negative integer`)
  }
}
