import { useState, useEffect } from 'react'
import { StatusRing } from './StatusRing'
import { useAppStore } from '../store/appStore'

export function StatusBar() {
  const { memoryCount, latency } = useAppStore()
  const [time, setTime] = useState(() => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))

  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    }, 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div
      className="flex items-center justify-between px-5 py-3 rounded-2xl"
      style={{
        background: '#3A0CA3',
        border: '3px solid #000000',
        boxShadow: '5px 5px 0px #000000',
      }}
    >
      {/* Left: ring + name */}
      <div className="flex items-center gap-3">
        <StatusRing />
        <span className="font-bold text-xl" style={{ color: '#ffffff', fontFamily: 'Inter, sans-serif', textShadow: '0 1px 4px rgba(0,0,0,0.15)' }}>
          Small O 🤖
        </span>
      </div>

      {/* Right: metadata */}
      <div
        className="flex items-center gap-4 text-xs font-semibold"
        style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(255,255,255,0.85)' }}
      >
        <span>phi3 · Ollama</span>
        {memoryCount > 0 && <span>🧠 {memoryCount}</span>}
        {latency > 0 && <span>⚡ {latency.toFixed(1)}s</span>}
        <span>{time}</span>
      </div>
    </div>
  )
}
