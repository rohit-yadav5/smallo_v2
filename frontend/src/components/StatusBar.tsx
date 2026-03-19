import { useAppStore } from '../store/appStore'

const STATE_COLORS: Record<string, string> = {
  idle: 'text-purple-400',
  listening: 'text-purple-300',
  thinking: 'text-yellow-400',
  speaking: 'text-green-400',
}

const STATE_LABELS: Record<string, string> = {
  idle: 'STANDBY',
  listening: 'LISTENING',
  thinking: 'PROCESSING',
  speaking: 'SPEAKING',
}

export function StatusBar() {
  const { voiceState, memoryCount, latency } = useAppStore()

  return (
    <div
      className="w-full flex items-center justify-between px-6 py-2"
      style={{
        background: 'rgba(0, 0, 0, 0.95)',
        fontFamily: 'Orbitron, sans-serif',
      }}
    >
      <div className="flex items-center gap-6">
        <span className="text-purple-300 font-bold tracking-widest text-sm">SMALL O</span>
        <span className="text-zinc-600 text-xs">CORE v1.0</span>
      </div>

      <div className="flex items-center gap-8 text-xs tracking-wider">
        <div className="flex items-center gap-2">
          <div
            className={`w-1.5 h-1.5 rounded-full ${voiceState === 'idle' ? 'bg-purple-500 opacity-50' : 'bg-purple-400 animate-pulse'}`}
          />
          <span className={STATE_COLORS[voiceState]}>{STATE_LABELS[voiceState]}</span>
        </div>

        <span className="text-zinc-600">
          MODEL: <span className="text-zinc-400">phi3 · Ollama</span>
        </span>

        <span className="text-zinc-600">
          MEMORY: <span className="text-purple-400">{memoryCount}</span>
        </span>

        {latency > 0 && (
          <span className="text-zinc-600">
            LATENCY: <span className="text-green-400">{latency.toFixed(1)}s</span>
          </span>
        )}
      </div>

      <div className="text-zinc-600 text-xs" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  )
}
