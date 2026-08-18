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

export const runIngest = (rebuild = false) =>
  request('/ingest', { method: 'POST', body: JSON.stringify({ rebuild }) })
