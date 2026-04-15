import { useState, useEffect } from 'react'
import { StatusRing } from './StatusRing'
import { useAppStore } from '../store/appStore'

// ── RAM pressure dot colours ──────────────────────────────────────────────────
const _PRESSURE_COLOR: Record<string, string> = {
  low:    '#4ade80',   // green
  medium: '#fbbf24',   // amber
  high:   '#f87171',   // red
}

// ── Inline SVG icons (12×12) ──────────────────────────────────────────────────

function MicIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <rect x="3.5" y="0.5" width="5" height="7" rx="2.5" fill="currentColor" />
      <path d="M1.5 6.5C1.5 9.0 3.5 10.5 6 10.5C8.5 10.5 10.5 9.0 10.5 6.5"
            stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <line x1="6" y1="10.5" x2="6" y2="12"
            stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  )
}

function SpeakerIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M1.5 4.5H3.5L6.5 2V10L3.5 7.5H1.5V4.5Z" fill="currentColor" />
      <path d="M8 4.3C8.9 5.1 8.9 6.9 8 7.7"
            stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M9.5 2.8C11 4.3 11 7.7 9.5 9.2"
            stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  )
}

function BrowserIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <rect x="0.6" y="1.6" width="10.8" height="8.8" rx="1.4"
            stroke="currentColor" strokeWidth="1.2" />
      <line x1="0.6" y1="4.5" x2="11.4" y2="4.5"
            stroke="currentColor" strokeWidth="1.2" />
      <circle cx="2.5" cy="3.0" r="0.65" fill="currentColor" />
      <circle cx="4.5" cy="3.0" r="0.65" fill="currentColor" />
    </svg>
  )
}

// ── Slash overlay (rendered on top of icon when inactive) ─────────────────────
function SlashOverlay() {
  return (
    <svg
      width="12" height="12" viewBox="0 0 12 12"
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      aria-hidden="true"
    >
      <line x1="2" y1="10" x2="10" y2="2"
            stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

// ── Pill toggle button ─────────────────────────────────────────────────────────

interface PillProps {
  active:   boolean
  onClick:  () => void
  icon:     React.ReactNode
  label:    string
  // Optional: "pressed" appearance distinct from active (used for BROWSER)
  pressed?: boolean
}

function TogglePill({ active, onClick, icon, label, pressed }: PillProps) {
  const isOn = active || pressed

  return (
    <button
      onClick={onClick}
      style={{
        display:         'inline-flex',
        alignItems:      'center',
        gap:             '4px',
        padding:         '3px 8px',
        borderRadius:    '999px',
        fontSize:        '11px',
        fontWeight:      500,
        cursor:          'pointer',
        userSelect:      'none',
        transition:      'background 0.15s, color 0.15s',
        fontFamily:      'JetBrains Mono, monospace',
        border:          isOn
          ? '1px solid rgba(147,197,253,0.4)'
          : '1px solid rgba(255,255,255,0.1)',
        background:      isOn
          ? 'rgba(96,165,250,0.22)'
          : 'rgba(255,255,255,0.07)',
        color:           isOn
          ? '#93c5fd'
          : 'rgba(255,255,255,0.38)',
        position:        'relative',
      }}
    >
      {/* Icon wrapper — relative so slash overlay can sit on top */}
      <span style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        {icon}
        {!active && !pressed && <SlashOverlay />}
      </span>
      <span style={{ textDecoration: 'none' }}>{label}</span>
    </button>
  )
}

// ── Component ──────────────────────────────────────────────────────────────────

export function StatusBar() {
  const {
    memoryCount, latency, availableRamGb, ramPressure, systemToast, clearSystemToast,
    sttEnabled, ttsEnabled, toggleSTT, toggleTTS,
    latestScreenshot, browserViewerOpen, setBrowserViewerOpen,
  } = useAppStore()

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
  const hasFeed  = latestScreenshot !== null

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

        {/* Right: metadata + pill toggles */}
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

          {/* ── Toggle pills: BROWSER · MIC · VOICE ─────────────────────── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <TogglePill
              active={hasFeed}
              pressed={browserViewerOpen}
              icon={<BrowserIcon />}
              label="BROWSER"
              onClick={() => setBrowserViewerOpen(!browserViewerOpen)}
            />
            <TogglePill
              active={sttEnabled}
              icon={<MicIcon />}
              label="MIC"
              onClick={toggleSTT}
            />
            <TogglePill
              active={ttsEnabled}
              icon={<SpeakerIcon />}
              label="VOICE"
              onClick={toggleTTS}
            />
          </div>

          {/* Clock — 12px gap via the parent gap-4 (16px) minus 6px pill gap = fine */}
          <span style={{ marginLeft: '6px' }}>{time}</span>
        </div>
      </div>
    </>
  )
}
