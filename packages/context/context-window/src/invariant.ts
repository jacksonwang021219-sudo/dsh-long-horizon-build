/**
 * Package-owned invariants for the sliding-window fold protocol.
 *
 * Every fold this plugin performs is a `compaction/prune` shadow-price event
 * immediately followed by the replacement it prices. Both halves are durable,
 * so the relation between them is checkable: the replacement must be this
 * plugin's own block, it must replace exactly the range the prune priced, and
 * it must cite every shadowed node. A fold that broke any of those would make
 * the token meter's surface accounting and every later replay wrong.
 *
 * @module @deepseek-ai/dsh-context-window/invariant
 */

import type { Context } from '@deepseek-ai/cordis'
import { isReplacementSurfaceEvent } from '@deepseek-ai/dsh-session'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { InvariantFailure, InvariantInstaller } from '@deepseek-ai/dsh-invariants'
import { WINDOW_SOURCE_PLUGIN } from './window.ts'

const PACKAGE_NAME = '@deepseek-ai/dsh-context-window'

/** Cordis companion plugin name. */
export const name = 'context-window-invariant'

/** Service required before the companion can reserve package ownership. */
export const inject = ['invariants']

/**
 * Whether a surface replacement is this plugin's own fold block. The source is
 * read structurally: a fold block is a plugin-sourced user message, and this
 * package is its only producer.
 */
function isWindowFold(event: SessionEvent): boolean {
  if (event.type !== 'user/message') return false
  const source = event.data.source
  return source.kind === 'plugin' && source.plugin === WINDOW_SOURCE_PLUGIN
}

/**
 * Validate one window fold: the prune that prices it is the event immediately
 * before it, and the pair must agree on the replaced range.
 */
function validateFold(
  prune: SessionEvent<'compaction/prune'>,
  replacement: SessionEvent,
  fail: InvariantFailure,
): void {
  if (!isReplacementSurfaceEvent(replacement)) {
    fail('a window fold block must replace the range its shadow-price event priced')
    return
  }
  const { shadowedRange, shadowedSeqs, shadowedTokenCount } = prune.data
  if (shadowedSeqs.length === 0) fail('window fold shadowedSeqs must be non-empty')
  if (shadowedSeqs[0] !== shadowedRange.start || shadowedSeqs.at(-1) !== shadowedRange.end) {
    fail('window fold shadowedRange must match the first and last shadowedSeqs')
  }
  if (!Number.isSafeInteger(shadowedTokenCount) || shadowedTokenCount < 0) {
    fail('window fold shadowedTokenCount must be a non-negative safe integer')
  }
  if (replacement.surfaceOp.start !== shadowedRange.start || replacement.surfaceOp.end !== shadowedRange.end) {
    fail('window fold block must replace exactly the range its shadow-price event priced')
  }
  const cited = new Set(replacement.sourceEventSeqs ?? [])
  for (const seq of shadowedSeqs) {
    if (!cited.has(seq)) fail(`window fold block does not cite shadowed node ${String(seq)}`)
  }
}

/**
 * Check every fold in one session's log. A `compaction/prune` is checked only
 * when the event right after it is this package's own fold block; the
 * compaction backend's checkpoint carries a different source and owns its own
 * companion, so the two protocols never validate each other's events.
 */
function validateSession(session: Session, fail: InvariantFailure): void {
  const events = session.events
  for (let index = 0; index < events.length; index += 1) {
    const event = events[index]
    if (event === undefined || event.type !== 'compaction/prune') continue
    const next = events[index + 1]
    if (next === undefined || !isWindowFold(next)) continue
    validateFold(event, next, fail)
  }
}

/** Install the fold-protocol checks over loaded and newly appended events. */
const install: InvariantInstaller = Object.assign((ctx: Context, fail: InvariantFailure) => {
  for (const session of ctx.sessions.list()) validateSession(session, fail)
  ctx.on('internal/dispatch', (_mode, eventName, args) => {
    if (eventName !== 'session/event') return
    const [session, event] = args as [Session, SessionEvent]
    if (!isWindowFold(event)) return
    const previous = session.events[event.seq - 1]
    if (previous === undefined || previous.type !== 'compaction/prune') {
      fail('a window fold block must be immediately preceded by its shadow-price event')
      return
    }
    validateFold(previous, event, fail)
  }, { global: true })
}, { inject: ['sessions'] })

/**
 * Register the context-window invariant companion.
 * @param ctx - Cordis context carrying the invariant service.
 * @returns the installed registration's disposer after setup succeeds.
 */
export const apply = (ctx: Context): Promise<() => void> =>
  Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
