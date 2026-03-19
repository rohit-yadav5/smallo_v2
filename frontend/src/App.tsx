import { StatusBar } from './components/StatusBar'
import { AICoreSphere } from './components/AICoreSphere'
import { ConversationStream } from './components/ConversationStream'
import { MemoryNetwork } from './components/MemoryNetwork'
import { SystemStats } from './components/SystemStats'
import { PluginNotifications } from './components/PluginNotifications'
import { ParticleBackground } from './components/ParticleBackground'
import { useWebSocket } from './hooks/useWebSocket'

export default function App() {
  useWebSocket()

  return (
    <div
      className="w-screen h-screen flex flex-col overflow-hidden"
      style={{ background: '#000000', color: '#e4e4e7' }}
    >
      <ParticleBackground />
      <PluginNotifications />

      {/* Status Bar */}
      <div className="relative z-10 shrink-0">
        <StatusBar />
        {/* Fade status bar into content below */}
        <div
          className="absolute left-0 right-0 bottom-0 translate-y-full pointer-events-none"
          style={{
            height: '32px',
            background: 'linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, transparent 100%)',
            zIndex: 20,
          }}
        />
      </div>

      {/* Main 3-column layout — no borders, panels bleed into background */}
      <div className="relative z-10 flex flex-1 overflow-hidden">

        {/* Left: Memory Network — fades rightward into center */}
        <div
          className="w-56 shrink-0 p-4"
          style={{
            background: 'linear-gradient(to right, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.4) 70%, transparent 100%)',
          }}
        >
          <MemoryNetwork />
        </div>

        {/* Center: AI Core + Conversation */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Sphere */}
          <div className="flex-shrink-0" style={{ height: '55%' }}>
            <AICoreSphere />
          </div>

          {/* Conversation stream — fades up into sphere with mask */}
          <div
            className="flex-1 overflow-hidden px-4 py-3"
            style={{
              background: 'linear-gradient(to bottom, transparent 0%, rgba(0,0,0,0.55) 25%, rgba(0,0,0,0.75) 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 18%)',
              maskImage: 'linear-gradient(to bottom, transparent 0%, black 18%)',
            }}
          >
            <ConversationStream />
          </div>
        </div>

        {/* Right: System Stats — fades leftward into center */}
        <div
          className="w-44 shrink-0 p-4"
          style={{
            background: 'linear-gradient(to left, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.4) 70%, transparent 100%)',
          }}
        >
          <SystemStats />
        </div>

      </div>
    </div>
  )
}
