/** AOP's bounded automation driver for the DeepSeek Harness headless profile. */

import { installModelSelection } from '@deepseek-ai/dsh-agent'

export const name = 'aop-headless-runner'
export const inject = ['agentDefaultModel', 'agents', 'sessions', 'headlessStartup']

function userMessage(text) {
  return Object.freeze({
    role: 'user',
    content: Object.freeze([Object.freeze({ type: 'text', text })]),
    source: Object.freeze({ kind: 'user' }),
    id: crypto.randomUUID(),
  })
}

function summarize(events, firstSeq) {
  let text = ''
  let reason
  const usage = {
    input_tokens: 0,
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    output_tokens: 0,
    reasoning_output_tokens: 0,
  }
  for (const event of events) {
    if (event.seq < firstSeq) continue
    if (event.type === 'assistant/message') {
      const current = event.data.message.content
        .filter(block => block.type === 'text')
        .map(block => block.text)
        .join('')
      if (current !== '') text = current
      const reported = event.data.usage ?? {}
      usage.input_tokens += reported.inputTokens ?? 0
      usage.cached_input_tokens += reported.cacheReadTokens ?? 0
      usage.cache_write_input_tokens += reported.cacheWriteTokens ?? 0
      usage.output_tokens += reported.outputTokens ?? 0
      usage.reasoning_output_tokens += reported.reasoningTokens ?? 0
    }
    if (event.type === 'turn/end') reason = event.data.reason
  }
  return { text, reason, usage }
}

function errorText(reason) {
  if (reason?.kind !== 'error') return undefined
  return reason.error?.message ?? reason.error?.code ?? 'DeepSeek Harness turn failed'
}

function write(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`)
}

async function run(ctx, task, sessionId, resume, exit) {
  await ctx.get('loader')?.await()
  const agents = ctx.get('agents')
  const sessions = ctx.get('sessions')
  const selection = ctx.get('agentDefaultModel')?.currentSelection()
  if (agents === undefined || sessions === undefined || selection === undefined) return

  const options = {
    agentOptions: { provider: selection.provider, model: selection.model },
    setup: (agentCtx) => {
      installModelSelection(agentCtx, { current: selection, assembled: undefined })
    },
  }
  const handle = resume
    ? await agents.resume({ ...options, resumeSessionId: sessionId })
    : await agents.create({ ...options, sessionId, meta: { cwd: process.cwd() } })
  const { agent } = handle
  await agent.whenIdle()
  const firstSeq = agent.session.seq
  write({ type: 'aop.dsh.started', session_id: agent.session.id })
  agent.followup(userMessage(task))
  await agent.whenIdle()
  await sessions.flush(agent.session)
  const outcome = summarize(agent.session.events, firstSeq)
  const error = errorText(outcome.reason)
  write({
    type: 'aop.dsh.result',
    session_id: agent.session.id,
    model: selection.model,
    final_message: outcome.text || null,
    usage: outcome.usage,
    completed: outcome.reason?.kind === 'completed',
    error: error ?? null,
  })
  exit(outcome.reason?.kind === 'completed' ? 0 : 1)
}

export function apply(ctx) {
  const exit = ctx.get('appExit')
  if (exit === undefined) throw new Error('aop-headless-runner requires the dsh launcher')
  const sessionId = process.env.AOP_DSH_SESSION_ID
  if (!sessionId) throw new Error('AOP_DSH_SESSION_ID is required')
  const task = ctx.get('headlessStartup')?.task
  if (!task) throw new Error('aop-headless-runner requires a task')
  void run(ctx, task, sessionId, process.env.AOP_DSH_RESUME === '1', exit)
    .catch((error) => {
      write({
        type: 'aop.dsh.result',
        session_id: sessionId,
        final_message: null,
        usage: {},
        completed: false,
        error: error instanceof Error ? error.message : String(error),
      })
      exit(1)
    })
}
