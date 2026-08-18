import { useEffect, useState } from 'react'
import { askQuestion, getHealth, runIngest } from './api.js'
import AskForm from './components/AskForm.jsx'
import AnswerPanel from './components/AnswerPanel.jsx'
import HistoryList from './components/HistoryList.jsx'

export default function App() {
  const [health, setHealth] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [ingesting, setIngesting] = useState(false)

  const refreshHealth = () => getHealth().then(setHealth).catch(() => setHealth(null))

  useEffect(() => {
    refreshHealth()
  }, [])

  async function handleAsk(question) {
    setLoading(true)
    setError(null)
    try {
      const data = await askQuestion(question)
      setResult(data)
      setHistory((previous) => [data, ...previous].slice(0, 20))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  async function handleIngest() {
    setIngesting(true)
    setError(null)
    try {
      const summary = await runIngest(false)
      await refreshHealth()
      setError(
        `Ingest done: ${summary.files_indexed} indexed, ${summary.files_skipped} unchanged, ` +
          `${summary.total_chunks} chunks total (${summary.elapsed_seconds}s)`,
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setIngesting(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Architecture Knowledge Base</h1>
          <p className="subtitle">
            Local RAG over public-domain architecture texts — answers grounded in
            retrieved passages only.
          </p>
        </div>
        <button className="ghost" onClick={handleIngest} disabled={ingesting}>
          {ingesting ? 'Ingesting…' : 'Re-ingest documents'}
        </button>
      </header>

      {health && (
        <div className={`status ${health.status}`}>
          <span className="dot" />
          {health.status} · {health.documents} documents · {health.chunks} chunks ·{' '}
          {health.dimension}-dim {health.embed_model} · {health.llm_model}
          {!health.ollama_reachable && ' · Ollama unreachable'}
        </div>
      )}

      <main>
        <div className="column">
          <AskForm onAsk={handleAsk} loading={loading} />
          <AnswerPanel result={result} error={error} loading={loading} />
        </div>
        <HistoryList history={history} onSelect={setResult} />
      </main>
    </div>
  )
}
