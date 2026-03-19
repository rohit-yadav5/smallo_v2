import { useState } from 'react'
import { StatusBar } from './components/StatusBar'
import { ConversationStream } from './components/ConversationStream'
import { InfoDrawer } from './components/InfoDrawer'
import { PluginNotifications } from './components/PluginNotifications'
import { useWebSocket } from './hooks/useWebSocket'
import { useAppStore } from './store/appStore'

const STATE_FOOTER: Record<string, { emoji: string; label: string }> = {
  idle:      { emoji: '',   label: '' },
  listening: { emoji: '🎙', label: 'Listening...' },
  thinking:  { emoji: '🤔', label: 'Thinking...' },
  speaking:  { emoji: '💬', label: 'Speaking...' },
}

export default function App() {
  useWebSocket()
  const voiceState = useAppStore((s) => s.voiceState)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const footer = STATE_FOOTER[voiceState]

  return (
    <div
      className="min-h-screen w-screen flex flex-col items-center"
      style={{ background: 'linear-gradient(135deg, #FFD60A 0%, #FF4D6D 50%, #7B2FBE 100%)' }}
    >
      <div className="w-full max-w-2xl flex flex-col px-4 py-4 gap-3" style={{ height: '100dvh' }}>

        {/* Status bar */}
        <StatusBar />

        {/* Conversation — scrollable, fills available space */}
        <ConversationStream />

        {/* Footer: voice state + drawer toggle */}
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

        {/* Collapsible info drawer */}
        <InfoDrawer open={drawerOpen} />

      </div>

      <PluginNotifications />
    </div>
  )
}
