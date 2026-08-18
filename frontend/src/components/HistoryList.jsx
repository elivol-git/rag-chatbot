export default function HistoryList({ history, onSelect }) {
  if (!history.length) return null

  return (
    <aside className="history">
      <h2>History</h2>
      <ul>
        {history.map((entry, index) => (
          <li key={`${entry.question}-${index}`}>
            <button onClick={() => onSelect(entry)}>
              <span className="history-question" dir="auto">
                {entry.question}
              </span>
              <span className="history-meta">
                {entry.chunks.length} chunks · {entry.total_ms} ms
              </span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
