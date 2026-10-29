import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [status, setStatus] = useState('Connecting...')
  const [classes, setClasses] = useState([])

  useEffect(() => {
    async function loadProject() {
      try {
        const [healthResponse, classesResponse] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/classes'),
        ])

        if (!healthResponse.ok || !classesResponse.ok) {
          throw new Error('API request failed')
        }

        const health = await healthResponse.json()
        const project = await classesResponse.json()

        setStatus(health.status)
        setClasses(project.classes)
      } catch {
        setStatus('API unavailable')
      }
    }

    loadProject()
  }, [])

  return (
    <main>
      <h1>Crop Disease Advisor</h1>
      <p>API status: {status}</p>

      <h2>Supported tomato leaf classes</h2>
      <ul>
        {classes.map((item) => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </main>
  )
}

export default App