const BASE = '/api'

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`)
  }
  return data
}

export const getHealth = () => request('/health')

export const askQuestion = (question, topK) =>
  request('/ask', {
    method: 'POST',
    body: JSON.stringify(topK ? { question, top_k: topK } : { question }),
  })

/**
 * Ask with Server-Sent Events. Calls onMeta once with the retrieved chunks,
 * onToken for each answer fragment, and onDone with the final timings.
 * EventSource cannot POST, so the stream is parsed off fetch() directly.
 */
export async function askQuestionStream(question, { onMeta, onToken, onDone, topK, signal }) {
  const response = await fetch(`${BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(topK ? { question, top_k: topK } : { question }),
    signal,
  })

  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.error || `Request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line.
    let split
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)

      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (!data) continue
      const payload = JSON.parse(data)

      if (event === 'meta') onMeta?.(payload)
      else if (event === 'token') onToken?.(payload.text)
      else if (event === 'done') onDone?.(payload)
      else if (event === 'error') throw new Error(payload.error)
    }
  }
}

export const runIngest = (rebuild = false) =>
  request('/ingest', { method: 'POST', body: JSON.stringify({ rebuild }) })
