/**
 * Per-session window state: what the window is waiting for, how many reminders
 * it already sent, and how much it has folded.
 *
 * The state is keyed by {@link Session} inside one mounted row, so several
 * sessions sharing a preset's standing composition stay apart.
 *
 * @module @deepseek-ai/dsh-context-window/state
 */

import type { Session } from '@deepseek-ai/dsh-session'
import type { WindowSummary } from './notes.ts'

/** A record the window is waiting for before it may fold a span. */
export interface PendingRecord {
  /** The turn the record must belong to. */
  readonly turn: number
  /** Inclusive first log seq the record must account for. */
  readonly start: number
  /** Inclusive last log seq the record must account for. */
  readonly end: number
}

/** A note already stored for one turn, ready to stand in for its trace. */
export interface StoredNoteRef {
  /** The stored note id, e.g. `task-0003`. */
  readonly noteId: string
  /** The four fields the window keeps visible. */
  readonly summary: WindowSummary
  /** Inclusive first log seq the note covers. */
  readonly start: number
  /** Inclusive last log seq the note covers. */
  readonly end: number
}

/** One recording that is running or finished, for the pseudo-async handshake. */
export interface RecordingHandle {
  /** The turn being recorded. */
  readonly turn: number
  /** Inclusive first log seq the record covers. */
  readonly start: number
  /** Inclusive last log seq the record covers. */
  readonly end: number
  /** Settles when the note is stored (or the recording gave up). */
  readonly done: Promise<StoredNoteRef | null>
}

/** Mutable window state for one session. */
export interface SessionWindowState {
  /** The record obligation the window injected and is waiting for. */
  pending: PendingRecord | undefined
  /** How many times the window has asked for {@link pending}. */
  pendingAttempts: number
  /** The recording running in the background, if any. */
  recording: RecordingHandle | undefined
  /** Notes stored so far, by turn. */
  notes: Map<number, StoredNoteRef>
  /** Turns a human request opened, in the main agent's own conversation. */
  readonly humanTurns: Set<number>
  /** The turn the reminder counter below belongs to. */
  reminderTurn: number
  /** Reminders the end-of-turn gate has already issued for {@link reminderTurn}. */
  reminders: number
  /** Turns whose record reminders ran out; the window stops folding their traces. */
  readonly abandoned: Set<number>
  /** Consecutive turns that ended without a usable record. */
  failures: number
  /** Set once the circuit breaker tripped: no more folding or recording. */
  frozen: boolean
  /** Whether the frozen notice has been injected. */
  announced: boolean
  /** How many folds this session has performed, for diagnostics. */
  folds: number
}

const states = new WeakMap<Session, SessionWindowState>()

/**
 * The window state of one session, created on first use.
 * @param session - the session to key by.
 * @returns its mutable window state.
 */
export function stateFor(session: Session): SessionWindowState {
  let state = states.get(session)
  if (state === undefined) {
    state = {
      pending: undefined,
      pendingAttempts: 0,
      recording: undefined,
      notes: new Map(),
      humanTurns: new Set(),
      reminderTurn: 0,
      reminders: 0,
      abandoned: new Set(),
      failures: 0,
      frozen: false,
      announced: false,
      folds: 0,
    }
    states.set(session, state)
  }
  return state
}

/**
 * Whether one session is a delegated child rather than the user's own
 * conversation.
 *
 * The window never runs for a child: a recorder forked from a parent inherits
 * the parent's composition, so without this guard every recording would spawn
 * the recording of a recording.
 * @param session - session to classify.
 * @returns true for any subagent session.
 */
export function isChildSession(session: Session): boolean {
  if (session.header.origin === 'subagent') return true
  return (session.header.delegationDepth ?? 0) > 0
}
