import { useEffect, useState } from 'react'
import './App.css'

const MAX_FILE_BYTES = 5 * 1024 * 1024
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

async function api(path, options = {}) {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(
      data?.detail || `Request failed with status ${response.status}`
    )
  }

  return data
}

function Sources({ sources }) {
  if (!sources?.length) return null

  return (
    <div className="sources">
      <h4>Sources</h4>
      <ul>
        {sources.map((source) => (
          <li key={`${source.reference}-${source.url}`}>
            [{source.reference}]{' '}
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

function App() {
  const [apiStatus, setApiStatus] = useState('Connecting')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [history, setHistory] = useState([])
  const [diagnosis, setDiagnosis] = useState(null)
  const [guidance, setGuidance] = useState(null)
  const [question, setQuestion] = useState('')
  const [answers, setAnswers] = useState([])
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [loadingGuidance, setLoadingGuidance] = useState(false)
  const [asking, setAsking] = useState(false)

  useEffect(() => {
    api('/api/health')
      .then(() => setApiStatus('Connected'))
      .catch(() => setApiStatus('Unavailable'))

    api('/api/diagnoses')
      .then(setHistory)
      .catch(() => setError('Could not load diagnosis history.'))
  }, [])

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  function handleFileChange(event) {
    const selected = event.target.files?.[0]
    setError('')

    if (!selected) {
      setFile(null)
      setPreview(null)
      return
    }

    if (!ALLOWED_TYPES.includes(selected.type)) {
      setError('Choose a JPEG, PNG, or WebP image.')
      setFile(null)
      setPreview(null)
      event.target.value = ''
      return
    }

    if (selected.size > MAX_FILE_BYTES) {
      setError('Choose an image smaller than 5 MB.')
      setFile(null)
      setPreview(null)
      event.target.value = ''
      return
    }

    setFile(selected)
    setPreview(URL.createObjectURL(selected))
  }

  async function loadGuidance(diagnosisId) {
    setLoadingGuidance(true)
    setGuidance(null)

    try {
      const result = await api(
        `/api/diagnoses/${diagnosisId}/guidance`,
        { method: 'POST' }
      )
      setGuidance(result)
    } catch (err) {
      setError(`Could not load guidance: ${err.message}`)
    } finally {
      setLoadingGuidance(false)
    }
  }

  async function handleUpload(event) {
    event.preventDefault()
    if (!file || uploading) return

    setUploading(true)
    setError('')
    setGuidance(null)
    setAnswers([])
    setDiagnosis(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const created = await api('/api/diagnoses', {
        method: 'POST',
        body: formData,
      })

      setDiagnosis(created)
      setHistory((current) => [
        created,
        ...current.filter((item) => item.id !== created.id),
      ].slice(0, 20))

      await loadGuidance(created.id)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function selectDiagnosis(item) {
    setDiagnosis(item)
    setFile(null)
    setPreview(null)
    setAnswers([])
    setQuestion('')
    setError('')
    await loadGuidance(item.id)
  }

  async function handleQuestion(event) {
    event.preventDefault()
    const text = question.trim()

    if (!diagnosis || text.length < 3 || asking) return

    setAsking(true)
    setError('')

    try {
      const result = await api(
        `/api/diagnoses/${diagnosis.id}/questions`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text }),
        }
      )

      setAnswers((current) => [
        ...current,
        { id: crypto.randomUUID(), question: text, ...result },
      ])
      setQuestion('')
    } catch (err) {
      setError(`Could not answer the question: ${err.message}`)
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div>
          <p className="eyebrow">Tomato leaf analysis</p>
          <h1>Crop Disease Advisor</h1>
          <p>Upload a tomato leaf photo and explore cited guidance.</p>
        </div>
        <span className={`api-status ${apiStatus === 'Connected' ? 'online' : ''}`}>
          API: {apiStatus}
        </span>
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      <div className="layout">
        <main>
          <section className="card">
            <h2>Analyze a leaf</h2>

            <form onSubmit={handleUpload}>
              <label htmlFor="leaf-image">Tomato leaf image</label>
              <input
                id="leaf-image"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
              />

              {preview && (
                <img
                  className="preview"
                  src={preview}
                  alt="Selected tomato leaf"
                />
              )}

              <p className="hint">JPEG, PNG, or WebP. Maximum 5 MB.</p>

              <button type="submit" disabled={!file || uploading}>
                {uploading ? 'Analyzing...' : 'Analyze image'}
              </button>
            </form>
          </section>

          {diagnosis && (
            <section className="card">
              <p className="eyebrow">Model prediction</p>
              <h2>{diagnosis.display_name}</h2>
              <p>
                Model score:{' '}
                <strong>
                  {(diagnosis.model_score * 100).toFixed(1)}%
                </strong>
              </p>
              <p className="hint">
                This is an image-model prediction, not a confirmed diagnosis.
              </p>

              <div className="divider" />

              <h3>Guidance</h3>
              {loadingGuidance ? (
                <p>Finding guidance...</p>
              ) : guidance ? (
                <>
                  <p className="answer-text">{guidance.answer}</p>
                  <Sources sources={guidance.sources} />
                </>
              ) : (
                <p>No guidance loaded.</p>
              )}
            </section>
          )}

          {diagnosis && (
            <section className="card">
              <h2>Ask a follow-up question</h2>

              <form onSubmit={handleQuestion}>
                <label htmlFor="question">Your question</label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="What symptoms should I check?"
                  maxLength={500}
                  rows={3}
                />
                <button
                  type="submit"
                  disabled={question.trim().length < 3 || asking}
                >
                  {asking ? 'Finding answer...' : 'Ask question'}
                </button>
              </form>

              {answers.map((item) => (
                <div className="question-result" key={item.id}>
                  <h3>{item.question}</h3>
                  <p className="answer-text">{item.answer}</p>
                  <Sources sources={item.sources} />
                </div>
              ))}
            </section>
          )}
        </main>

        <aside className="card history">
          <h2>Recent diagnoses</h2>

          {history.length === 0 ? (
            <p className="hint">No diagnoses yet.</p>
          ) : (
            <ul>
              {history.map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    className={
                      item.id === diagnosis?.id
                        ? 'history-item active'
                        : 'history-item'
                    }
                    onClick={() => selectDiagnosis(item)}
                  >
                    <strong>{item.display_name}</strong>
                    <span>
                      {new Date(item.created_at).toLocaleString()}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </div>
  )
}

export default App