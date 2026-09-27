/**
 * The system-prompt section that teaches the main agent the one thing it needs
 * to know about the window: how to read back what left it, and how to ask for a
 * correction.
 *
 * The agent is deliberately given no way to write notes — no tool, no
 * instruction, no hint that writing is possible. Recording is the harness's
 * job, performed by a forked note keeper that sees the same conversation; what
 * remains here is the read side.
 *
 * @module @deepseek-ai/dsh-context-window/prompt
 */

import type { ResolvedContextWindowConfig } from './config.ts'
import { LOG_TOOL_NAME, NOTE_FIX_TOOL_NAME, NOTE_TOOL_NAME } from './tools.ts'

/** Section name of the sliding-window protocol in the system prompt. */
export const WINDOW_POLICY_SECTION = 'context:sliding-window'

/** Prompt order of that section: after tool guidance, before the tail sections. */
export const WINDOW_POLICY_ORDER = 190

/**
 * Render the read-side protocol for one mounted row.
 * @param config - the validated policy of the mounted row.
 * @returns the section text.
 */
export function windowPolicyText(config: ResolvedContextWindowConfig): string {
  const turns = String(config.retainUserTurns)
  const retain = String(Math.round(config.retainTurnTokens / 1000))
  const trigger = String(Math.round(config.foldTriggerTokens / 1000))
  return [
    '## Session memory',
    '',
    `This session keeps a rolling window over its history: the ${turns} most recent requests stay verbatim, the traces that answered older ones are replaced by notes the harness keeps for them, and a single request's own trace is condensed once it passes ${trigger}K tokens (down to the most recent ${retain}K).`,
    '',
    'Writing those notes is not your job and is not something you can do — the harness records them from the session itself. Your side of the contract:',
    '',
    `1. **Read before you guess.** When something you need is no longer in your visible history, read it back: \`${NOTE_TOOL_NAME}\` lists the notes kept for this session and returns any one of them, and \`${LOG_TOOL_NAME}\` returns the original events for a seq range a note cites. Never invent a fact that a note or the log can settle, and never redo work whose result the notes already record.`,
    `2. **Correct what is wrong.** A note is a claim, not a fact; if one is wrong or stale, say so precisely with \`${NOTE_FIX_TOOL_NAME}\`: name the note, what exactly is wrong, how it should read, and the evidence (a log seq range, a path, a command result). The note keeper audits your request and records its verdict, so a rejected correction stays visible too.`,
    '3. **Treat summaries as background.** Blocks framed as `<turn-summary>` or `<folded-history>` are notes the harness wrote, standing where the traces they replaced used to be. Verify anything you are about to depend on, prefer what the workspace says now, and do not acknowledge or restate them.',
    '',
    'Nothing is lost when a trace leaves the window: the session log is append-only and complete, and every note states the log range it covers.',
  ].join('\n')
}
