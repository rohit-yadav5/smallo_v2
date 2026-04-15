import { useState, useRef, useCallback } from 'react'
import { StatusBar } from './components/StatusBar'
import { ConversationStream } from './components/ConversationStream'
import { InfoDrawer } from './components/InfoDrawer'
import { PluginNotifications } from './components/PluginNotifications'
import { PlanProgress } from './components/PlanProgress'
import { WebViewer } from './components/WebViewer'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useWebSocket } from './hooks/useWebSocket'
import { useMicrophone } from './hooks/useMicrophone'
import { useAppStore } from './store/appStore'

/* ── Cartoon doodles — left side ─────────────────────────────── */
function LeftDeco() {
  return (
    <div className="side-deco" style={{ position: 'fixed', top: 0, left: 0, width: 'calc((100vw - 680px) / 2)', height: '100%', pointerEvents: 'none', userSelect: 'none', overflow: 'hidden' }}>

      {/* Big 5-point star */}
      <svg width="88" height="88" viewBox="0 0 100 100"
        style={{ position: 'absolute', top: '6%', left: '18%', transform: 'rotate(-18deg)' }}>
        <polygon points="50,5 61,36 95,36 68,57 79,91 50,70 21,91 32,57 5,36 39,36"
          fill="#FFD60A" stroke="#000" strokeWidth="5" strokeLinejoin="round" />
      </svg>

      {/* Zigzag line */}
      <svg width="110" height="36" viewBox="0 0 110 36"
        style={{ position: 'absolute', top: '26%', left: '5%', transform: 'rotate(-4deg)' }}>
        <polyline points="5,32 25,4 45,32 65,4 85,32 105,4"
          fill="none" stroke="#3A0CA3" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>

      {/* Lightning bolt */}
      <svg width="52" height="88" viewBox="0 0 52 88"
        style={{ position: 'absolute', top: '40%', left: '28%', transform: 'rotate(8deg)' }}>
        <polygon points="36,0 12,50 28,50 16,88 48,36 30,36 44,0"
          fill="#FF6B35" stroke="#000" strokeWidth="4" strokeLinejoin="round" />
      </svg>

      {/* Dotted trio */}
      <svg width="96" height="60" viewBox="0 0 96 60"
        style={{ position: 'absolute', top: '62%', left: '8%' }}>
        <circle cx="18" cy="38" r="16" fill="#4CC9F0" stroke="#000" strokeWidth="4" />
        <circle cx="52" cy="20" r="12" fill="#F72585" stroke="#000" strokeWidth="3" />
        <circle cx="80" cy="42" r="10" fill="#7B2FBE" stroke="#000" strokeWidth="3" />
      </svg>

      {/* Small diamond */}
      <svg width="54" height="54" viewBox="0 0 100 100"
        style={{ position: 'absolute', top: '83%', left: '22%', transform: 'rotate(12deg)' }}>
        <polygon points="50,5 95,50 50,95 5,50"
          fill="#4361EE" stroke="#000" strokeWidth="5" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

/* ── Cartoon doodles — right side ────────────────────────────── */
function RightDeco() {
  return (
    <div className="side-deco" style={{ position: 'fixed', top: 0, right: 0, width: 'calc((100vw - 680px) / 2)', height: '100%', pointerEvents: 'none', userSelect: 'none', overflow: 'hidden' }}>

      {/* Wavy squiggle */}
      <svg width="110" height="46" viewBox="0 0 110 46"
        style={{ position: 'absolute', top: '7%', right: '10%', transform: 'rotate(6deg)' }}>
        <path d="M5,38 Q18,5 32,38 Q46,5 60,38 Q74,5 88,38 Q101,5 110,38"
          fill="none" stroke="#FF4D6D" strokeWidth="6" strokeLinecap="round" />
      </svg>

      {/* Starburst / 8-point star */}
      <svg width="88" height="88" viewBox="0 0 100 100"
        style={{ position: 'absolute', top: '22%', right: '18%', transform: 'rotate(22deg)' }}>
        <polygon points="50,5 58,38 90,26 70,54 95,72 62,68 60,100 44,70 10,80 30,55 8,35 40,40"
          fill="#FF4D6D" stroke="#000" strokeWidth="4" strokeLinejoin="round" />
      </svg>

      {/* Speech bubble */}
      <svg width="88" height="82" viewBox="0 0 100 92"
        style={{ position: 'absolute', top: '44%', right: '12%', transform: 'rotate(-10deg)' }}>
        <path d="M10,10 Q10,4 16,4 L84,4 Q90,4 90,10 L90,60 Q90,66 84,66 L56,66 L46,84 L40,66 L16,66 Q10,66 10,60 Z"
          fill="#4361EE" stroke="#000" strokeWidth="4" strokeLinejoin="round" />
        <circle cx="32" cy="36" r="6" fill="#FFD60A" />
        <circle cx="50" cy="36" r="6" fill="#FFD60A" />
        <circle cx="68" cy="36" r="6" fill="#FFD60A" />
      </svg>

      {/* Small dots row */}
      <svg width="80" height="22" viewBox="0 0 80 22"
        style={{ position: 'absolute', top: '68%', right: '14%', transform: 'rotate(-5deg)' }}>
        <circle cx="11" cy="11" r="9" fill="#FFD60A" stroke="#000" strokeWidth="3" />
        <circle cx="40" cy="11" r="7" fill="#FF6B35" stroke="#000" strokeWidth="3" />
        <circle cx="65" cy="11" r="9" fill="#4CC9F0" stroke="#000" strokeWidth="3" />
      </svg>

      {/* Diamond */}
      <svg width="58" height="58" viewBox="0 0 100 100"
        style={{ position: 'absolute', top: '82%', right: '22%', transform: 'rotate(-14deg)' }}>
        <polygon points="50,5 95,50 50,95 5,50"
          fill="#FFD60A" stroke="#000" strokeWidth="5" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

const STATE_FOOTER: Record<string, { emoji: string; label: string }> = {
  idle:      { emoji: '',   label: '' },
  listening: { emoji: '🎙', label: 'Listening...' },
  thinking:  { emoji: '🤔', label: 'Thinking...' },
  speaking:  { emoji: '💬', label: 'Speaking...' },
}

export default function App() {
  const { sendTextInput }  = useWebSocket()
  const { startMic }       = useMicrophone()
  const voiceState         = useAppStore((s) => s.voiceState)
  const wsConnected        = useAppStore((s) => s.wsConnected)
  const browserViewerOpen  = useAppStore((s) => s.browserViewerOpen)
  const [drawerOpen, setDrawerOpen]   = useState(false)
  const [inputText,  setInputText]    = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const footer = STATE_FOOTER[voiceState]

  const handleSend = useCallback(() => {
    const text = inputText.trim()
    if (!text || !wsConnected) return
    sendTextInput(text)
    setInputText('')
    inputRef.current?.focus()
  }, [inputText, wsConnected, sendTextInput])

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <ErrorBoundary>
      <div
        className="min-h-screen w-screen flex flex-col items-center"
        style={{ background: 'linear-gradient(135deg, #FFD60A 0%, #FF4D6D 50%, #7B2FBE 100%)' }}
      >
        <LeftDeco />
        <RightDeco />
        <div className="w-full max-w-2xl flex flex-col px-4 py-4 gap-3" style={{ height: '100dvh' }}>
          <StatusBar />
          <ConversationStream onStartMic={startMic} />
          <PlanProgress />
          {browserViewerOpen && <WebViewer />}

          {/* ── Text input bar — always visible ─────────────────────── */}
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-xl shrink-0"
            style={{
              background:  '#F8F0FF',
              border:      '3px solid #000',
              boxShadow:   '5px 5px 0px #000',
            }}
          >
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message…"
              disabled={!wsConnected}
              className="flex-1 bg-transparent outline-none text-sm font-medium placeholder:text-gray-400"
              style={{
                fontFamily: 'Inter, sans-serif',
                color: '#000',
              }}
            />
            <button
              onClick={handleSend}
              disabled={!inputText.trim() || !wsConnected}
              className="shrink-0 px-3 py-1 rounded-lg text-sm font-black transition-all active:translate-y-0.5 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                background:  '#7B2FBE',
                border:      '2px solid #000',
                boxShadow:   '3px 3px 0px #000',
                color:       '#FFD60A',
                cursor:      'pointer',
              }}
            >
              Send ↵
            </button>
          </div>

          {/* ── Footer status + drawer toggle ────────────────────────── */}
          <div
            className="flex items-center justify-between px-4 py-2.5 rounded-xl shrink-0"
            style={{
              background: '#FF6B35',
              border: '3px solid #000000',
              boxShadow: '5px 5px 0px #000000',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <span className="text-sm font-semibold flex items-center gap-1.5" style={{ color: '#ffffff' }}>
              {footer.emoji && <span>{footer.emoji}</span>}
              {footer.label || <span style={{ color: 'rgba(255,255,255,0.7)' }}>Ready ✨</span>}
            </span>
            <button
              className="text-xs flex items-center gap-1 px-3 py-1 rounded-lg transition-all active:translate-y-0.5"
              style={{
                fontFamily: 'JetBrains Mono, monospace',
                color: '#000000',
                background: '#FFD60A',
                border: '2px solid #000000',
                boxShadow: '3px 3px 0px #000000',
                cursor: 'pointer',
                fontWeight: 700,
              }}
              onClick={() => setDrawerOpen((o) => !o)}
            >
              <span>{drawerOpen ? '⌃' : '⌄'}</span>
              <span>Memory · Stats</span>
            </button>
          </div>
          <InfoDrawer open={drawerOpen} />
        </div>
        <PluginNotifications />
      </div>
    </ErrorBoundary>
  )
}
