import SourceList from './SourceList.jsx'

export default function AnswerPanel({ result, error, loading }) {
  // While streaming, `result` already holds the retrieved context and the
  // answer so far, so only the pre-retrieval moment shows a placeholder.
  if (loading && !result) {
    return (
      <div className="panel">
        <p className="muted">Retrieving context…</p>
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
      {/* dir="auto" lets each block pick its own direction from its first
          strong character, so Hebrew renders RTL and English LTR. */}
      <p className="question" dir="auto">
        {result.question}
      </p>
      <p className={`answer${result.grounded ? '' : ' ungrounded'}`} dir="auto">
        {result.answer}
        {result.streaming && <span className="caret" />}
      </p>

      <div className="metrics">
        <span>retrieval {result.retrieval_ms} ms</span>
        {result.streaming ? (
          <span>generating…</span>
        ) : (
          <>
            <span>llm {result.llm_ms} ms</span>
            <span>total {result.total_ms} ms</span>
          </>
        )}
        <span>{result.chunks.length} chunks</span>
        {!result.grounded && !result.streaming && (
          <span className="warn">no context above threshold</span>
        )}
      </div>

      <SourceList chunks={result.chunks} />
    </div>
  )
}
