import { useState } from 'react'

const SAMPLES = [
  "What are Vitruvius' three principles of good architecture?",
  'What defines Brutalist architecture?',
  'How does a flying buttress work?',
  'What did the Bauhaus teach?',
]

export default function AskForm({ onAsk, loading }) {
  const [question, setQuestion] = useState('')

  function submit(event) {
    event.preventDefault()
    const trimmed = question.trim()
    if (trimmed && !loading) onAsk(trimmed)
  }

  return (
    <form className="ask-form" onSubmit={submit}>
      <div className="ask-row">
        <input
          type="text"
          value={question}
          placeholder="Ask about architecture…"
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </div>
      <div className="samples">
        {SAMPLES.map((sample) => (
          <button
            key={sample}
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => {
              setQuestion(sample)
              onAsk(sample)
            }}
          >
            {sample}
          </button>
        ))}
      </div>
    </form>
  )
}
