import { useState } from 'react'

export default function SourceList({ chunks }) {
  const [open, setOpen] = useState(false)

  if (!chunks?.length) return null

  return (
    <section className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Retrieved context ({chunks.length} chunk
        {chunks.length === 1 ? '' : 's'})
      </button>

      {open && (
        <ol className="source-list">
          {chunks.map((chunk, index) => (
            <li key={`${chunk.source}-${chunk.chunk_index}`}>
              <div className="source-head">
                <span className="source-index">[{index + 1}]</span>
                <span className="source-title">{chunk.title}</span>
                <span className="source-file">{chunk.source}</span>
                <span className="score" title="cosine similarity">
                  {chunk.score.toFixed(3)}
                </span>
              </div>
              <div className="score-bar">
                <div style={{ width: `${Math.max(0, Math.min(1, chunk.score)) * 100}%` }} />
              </div>
              <p className="source-text" dir="auto">
                {chunk.text}
              </p>
              {chunk.source_url && (
                <a href={chunk.source_url} target="_blank" rel="noreferrer">
                  original source
                </a>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
