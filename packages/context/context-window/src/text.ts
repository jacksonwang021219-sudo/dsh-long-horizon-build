/**
 * Small text helpers shared by the window, its recorder, and its tools.
 *
 * @module @deepseek-ai/dsh-context-window/text
 */

import type { ContentBlock } from '@deepseek-ai/dsh-llm'

/** Join the text blocks of one content array, dropping non-text blocks. */
export function contentTextOf(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type !== 'text') continue
    const text = block.text.trim()
    if (text.length > 0) parts.push(text)
  }
  return parts.join('\n\n').trim()
}

/** Render an object as canonical JSON: sorted keys, two-space indent, newline. */
export function canonicalJson(value: unknown): string {
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
