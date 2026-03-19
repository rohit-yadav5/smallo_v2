import { useAppStore } from '../store/appStore'

const RING_STYLES: Record<string, { bg: string; shadow: string; animate: string }> = {
  idle:      { bg: '#d1c9be', shadow: 'none',                              animate: '' },
  listening: { bg: '#3b82f6', shadow: '0 0 8px rgba(59,130,246,0.6)',      animate: 'animate-pulse' },
  thinking:  { bg: '#f59e0b', shadow: '0 0 8px rgba(245,158,11,0.6)',      animate: 'animate-ping-slow' },
  speaking:  { bg: '#10b981', shadow: '0 0 10px rgba(16,185,129,0.7)',     animate: 'animate-pulse' },
}

export function StatusRing() {
  const voiceState = useAppStore((s) => s.voiceState)
  const style = RING_STYLES[voiceState] ?? RING_STYLES.idle

  return (
    <div className="relative flex items-center justify-center" style={{ width: 20, height: 20 }}>
      {/* Outer pulse ring */}
      {voiceState !== 'idle' && (
        <div
          className="absolute rounded-full animate-ping"
          style={{
            width: 20,
            height: 20,
            background: style.bg,
            opacity: 0.3,
            animationDuration: voiceState === 'thinking' ? '0.7s' : '1.2s',
          }}
        />
      )}
      {/* Inner dot */}
      <div
        className="rounded-full"
        style={{
          width: 12,
          height: 12,
          background: style.bg,
          boxShadow: style.shadow,
          transition: 'background 0.4s ease, box-shadow 0.4s ease',
        }}
      />
    </div>
  )
}
