#!/usr/bin/env node
/**
 * Decode one DeepSeek Harness JSONL session artifact to stdout.
 *
 * The artifact is a concatenated-frame Zstandard container whose first frame is
 * exactly the header line. Frames are decoded independently, so a torn final
 * frame costs only its own batch.
 *
 *   node readlog.mjs <session.jsonl.zstd> [out.jsonl]
 */

import { zstdDecompressSync } from 'node:zlib'
import { readFileSync, writeFileSync } from 'node:fs'

const MAGIC = Buffer.from([0x28, 0xb5, 0x2f, 0xfd])

/** Split a concatenated-frame buffer at each frame magic. */
function frameBounds(input) {
  const starts = []
  let index = input.indexOf(MAGIC, 0)
  while (index !== -1) {
    starts.push(index)
    index = input.indexOf(MAGIC, index + 4)
  }
  return starts.map((start, position) => ({
    start,
    end: position + 1 < starts.length ? starts[position + 1] : input.length,
  }))
}

const source = process.argv[2]
if (source === undefined) {
  process.stderr.write('usage: readlog.mjs <session.jsonl.zstd> [out.jsonl]\n')
  process.exit(2)
}
const input = readFileSync(source)
const chunks = []
for (const [position, bound] of frameBounds(input).entries()) {
  try {
    chunks.push(zstdDecompressSync(input.subarray(bound.start, bound.end)))
  } catch (error) {
    process.stderr.write(`[frame ${position} undecodable: ${error.message}]\n`)
  }
}
const text = Buffer.concat(chunks).toString('utf8')
if (process.argv[3] === undefined) process.stdout.write(text)
else writeFileSync(process.argv[3], text)
