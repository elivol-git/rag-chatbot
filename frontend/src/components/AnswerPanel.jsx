import SourceList from './SourceList.jsx'

export default function AnswerPanel({ result, error, loading }) {
  if (loading) {
    return (
      <div className="panel">
        <p className="muted">Retrieving context and generating a grounded answer…</p>
      </div>
    )
  }
  if (error) {
    return (
      <div className="panel error">
        <strong>Error:</strong> {error}
      </div>
    )
  }
  if (!result) return null

  return (
    <div className="panel">
      <p className="question">{result.question}</p>
      <p className={`answer${result.grounded ? '' : ' ungrounded'}`}>{result.answer}</p>

      <div className="metrics">
        <span>retrieval {result.retrieval_ms} ms</span>
        <span>llm {result.llm_ms} ms</span>
        <span>total {result.total_ms} ms</span>
        <span>{result.chunks.length} chunks</span>
        {!result.grounded && <span className="warn">no context above threshold</span>}
      </div>

      <SourceList chunks={result.chunks} />
    </div>
  )
}
