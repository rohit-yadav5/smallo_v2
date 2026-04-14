import { useState, useEffect } from 'react'
import { StatusRing } from './StatusRing'
import { useAppStore } from '../store/appStore'

// ── RAM pressure dot colours ──────────────────────────────────────────────────
const _PRESSURE_COLOR: Record<string, string> = {
  low:    '#4ade80',   // green
  medium: '#fbbf24',   // amber
  high:   '#f87171',   // red
}

export function StatusBar() {
  const { memoryCount, latency, availableRamGb, ramPressure, systemToast, clearSystemToast } =
    useAppStore()
  const [time, setTime] = useState(() =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  )

  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    }, 30_000)
    return () => clearInterval(id)
  }, [])

  const dotColor = ramPressure ? _PRESSURE_COLOR[ramPressure] : 'rgba(255,255,255,0.3)'

  return (
    <>
      {/* ── System toast (low_memory / model_swap) ─────────────────────── */}
      {systemToast && (
        <div
          onClick={clearSystemToast}
          style={{
            position:     'fixed',
            top:          '12px',
            right:        '12px',
            zIndex:       9999,
            background:   systemToast.kind === 'warning'
              ? 'rgba(251,191,36,0.95)'
              : 'rgba(100,116,139,0.95)',
            color:        systemToast.kind === 'warning' ? '#000' : '#fff',
            fontFamily:   'JetBrains Mono, monospace',
            fontSize:     '11px',
            fontWeight:   '700',
            padding:      '6px 12px',
            borderRadius: '8px',
            border:       '2px solid rgba(0,0,0,0.2)',
            boxShadow:    '2px 2px 0 rgba(0,0,0,0.15)',
            cursor:       'pointer',
            maxWidth:     '280px',
          }}
        >
          {systemToast.kind === 'warning' ? '⚠ ' : 'ℹ '}{systemToast.message}
        </div>
      )}

      {/* ── Status bar ───────────────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-5 py-3 rounded-2xl"
        style={{
          background: '#3A0CA3',
          border:     '3px solid #000000',
          boxShadow:  '5px 5px 0px #000000',
        }}
      >
        {/* Left: ring + name */}
        <div className="flex items-center gap-3">
          <StatusRing />
          <span
            className="font-bold text-xl"
            style={{
              color:      '#ffffff',
              fontFamily: 'Inter, sans-serif',
              textShadow: '0 1px 4px rgba(0,0,0,0.15)',
            }}
          >
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

          {/* RAM indicator — dot + GB number */}
          {availableRamGb !== null && (
            <span
              title={`Available RAM: ${availableRamGb.toFixed(1)} GB — pressure: ${ramPressure ?? '?'}`}
              style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              {/* Pressure dot */}
              <span
                style={{
                  display:      'inline-block',
                  width:        '7px',
                  height:       '7px',
                  borderRadius: '50%',
                  background:   dotColor,
                  flexShrink:   0,
                }}
              />
              {availableRamGb.toFixed(1)}GB
            </span>
          )}

          <span>{time}</span>
        </div>
      </div>
    </>
  )
}
