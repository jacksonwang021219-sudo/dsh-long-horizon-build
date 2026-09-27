/**
 * UltraHorizon environment bridge: register the episode's environment actions
 * as ordinary DSH tools.
 *
 * The benchmark's own loop gives the agent tools (`input_sequences`, `move`,
 * `commit_final_result`, …) and the environment's observations come back as
 * tool results inside ONE agent task — not as a sequence of user turns. This
 * plugin reproduces that mapping inside DSH: it asks the Python bridge (which
 * owns the environment) for the action list, registers each action as a tool,
 * and forwards every call to it.
 *
 * Configuration comes from the preset row:
 *
 * ```yaml
 * - id: uh-bridge
 *   name: '/abs/path/uh_bridge_plugin.mjs'
 *   config:
 *     url: 'http://127.0.0.1:PORT'
 * ```
 *
 * @module uh-bridge
 */

/** One tool description served by the Python bridge. */
const TOOL_OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: { observation: { type: 'string' } },
    required: ['observation'],
  },
  render: (_args, value) => [{ type: 'text', text: String(value.observation ?? '') }],
}

export default {
  name: 'uh-bridge',
  inject: ['tools'],

  async apply(ctx, config) {
    const base = String(config?.url ?? '')
    if (base === '') throw new Error('uh-bridge: config.url is required')
    const response = await fetch(`${base}/tools`)
    if (!response.ok) throw new Error(`uh-bridge: ${base}/tools answered ${response.status}`)
    const payload = await response.json()
    const actions = Array.isArray(payload.actions) ? payload.actions : []
    if (actions.length === 0) throw new Error('uh-bridge: the environment advertised no actions')

    for (const action of actions) {
      ctx.tools.register({
        name: action.name,
        description: action.description,
        parameters: action.parameters,
        output: TOOL_OUTPUT,
        execute: async (args) => {
          const called = await fetch(`${base}/call`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ name: action.name, args }),
          })
          const result = await called.json()
          if (result?.isError === true) {
            throw new Error(String(result.error ?? `${action.name} failed`))
          }
          return { observation: JSON.stringify(result?.observation ?? null, null, 2) }
        },
      })
    }
    ctx.logger.info(`uh-bridge: registered ${actions.length} environment tool(s) from ${base}`)
  },
}
