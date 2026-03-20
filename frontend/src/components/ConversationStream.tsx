import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

const PLUGIN_ICONS: Record<string, string> = {
  web:      '🌐',
  internet: '🌐',
  computer: '💻',
  security: '🔒',
}

function fmt(secs: number) {
  return `${secs.toFixed(2)}s`
}

// ── Countdown colours cycling through the pop palette ────────────────
const COUNT_COLORS = ['#FF4D6D', '#FF6B35', '#FFD60A', '#4CC9F0', '#7B2FBE', '#4361EE']

function EmptyState() {
  const voiceState  = useAppStore((s) => s.voiceState)
  const wsConnected = useAppStore((s) => s.wsConnected)
  const [count, setCount]   = useState(5)
  const [phase, setPhase]   = useState<'countdown' | 'waiting' | 'ready'>('countdown')

  // Tick down 5 → 0
  useEffect(() => {
    if (phase !== 'countdown') return
    if (count <= 0) { setPhase('waiting'); return }
    const t = setTimeout(() => setCount((c) => c - 1), 1000)
    return () => clearTimeout(t)
  }, [count, phase])

  // Bot signals it is ready
  useEffect(() => {
    if (voiceState === 'listening' || voiceState === 'thinking' || voiceState === 'speaking') {
      setPhase('ready')
    }
  }, [voiceState])

  // ── Disconnected ──────────────────────────────────────────────────────
  if (!wsConnected) {
    return (
      <motion.div
        className="flex-1 flex items-center justify-center"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      >
        <div className="text-center flex flex-col items-center gap-4">
          <div
            className="w-20 h-20 rounded-2xl flex items-center justify-center text-4xl"
            style={{ background: '#FF4D6D', border: '4px solid #000', boxShadow: '6px 6px 0px #000' }}
          >
            ✕
          </div>
          <p className="text-xl font-black" style={{ color: '#000' }}>
            Please wait while the server starts
          </p>
          <div className="flex gap-1.5 mt-2">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: '#FF4D6D', border: '2px solid #000' }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.4 }}
              />
            ))}
          </div>
        </div>
      </motion.div>
    )
  }

  // ── Ready — voiceState flipped to listening ───────────────────────────
  if (phase === 'ready' || voiceState === 'listening') {
    return (
      <motion.div
        className="flex-1 flex items-center justify-center"
        initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        <div className="text-center flex flex-col items-center gap-3">
          <motion.div
            className="text-5xl"
            animate={{ rotate: [0, -10, 10, -10, 0] }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            🎤
          </motion.div>
          <div
            className="px-5 py-2 rounded-xl"
            style={{ background: '#4CC9F0', border: '3px solid #000', boxShadow: '5px 5px 0px #000' }}
          >
            <p className="text-base font-black" style={{ color: '#000' }}>Say something!</p>
          </div>
        </div>
      </motion.div>
    )
  }

  // ── Counting down ─────────────────────────────────────────────────────
  const displayCount = count > 0 ? count : '…'
  const color = COUNT_COLORS[(5 - count) % COUNT_COLORS.length]

  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center flex flex-col items-center gap-5">

        {/* Label */}
        <div
          className="px-4 py-1.5 rounded-xl"
          style={{ background: '#3A0CA3', border: '3px solid #000', boxShadow: '4px 4px 0px #000' }}
        >
          <p className="text-sm font-black tracking-widest uppercase" style={{ color: '#FFD60A' }}>
            Starting up
          </p>
        </div>

        {/* Big animated number */}
        <AnimatePresence mode="wait">
          <motion.div
            key={displayCount}
            initial={{ scale: 1.6, opacity: 0 }}
            animate={{ scale: 1,   opacity: 1 }}
            exit={{    scale: 0.4, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 400, damping: 22 }}
            className="w-28 h-28 rounded-3xl flex items-center justify-center"
            style={{
              background:  color,
              border:      '5px solid #000',
              boxShadow:   '8px 8px 0px #000',
              fontSize:    '4rem',
              fontWeight:  900,
              color:       '#000',
              fontFamily:  'Inter, sans-serif',
            }}
          >
            {displayCount}
          </motion.div>
        </AnimatePresence>

        {/* Progress dots */}
        <div className="flex gap-2">
          {[5, 4, 3, 2, 1].map((n) => (
            <div
              key={n}
              className="w-3 h-3 rounded-full"
              style={{
                background: n >= count && count > 0 ? '#000' : 'rgba(0,0,0,0.2)',
                border: '2px solid rgba(0,0,0,0.3)',
                transition: 'background 0.3s',
              }}
            />
          ))}
        </div>

        <p className="text-xs font-semibold" style={{ color: 'rgba(0,0,0,0.4)' }}>
          Loading models...
        </p>
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────
export function ConversationStream() {
  const messages  = useAppStore((s) => s.messages)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div
      className="flex flex-col gap-4 overflow-y-auto flex-1 px-1 py-2"
      style={{ fontFamily: 'Inter, sans-serif' }}
    >
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.22, type: 'spring', stiffness: 300, damping: 24 }}
          >
            {msg.role === 'user' ? (
              <div className="flex flex-col items-end gap-1">
                <div
                  className="rounded-2xl rounded-br-sm px-4 py-3 max-w-[85%]"
                  style={{ background: '#FFD60A', border: '3px solid #000', boxShadow: '5px 5px 0px #000' }}
                >
                  <p className="text-base leading-relaxed" style={{ color: '#000', fontWeight: 600 }}>
                    {msg.text}
                  </p>
                </div>
                {msg.stt_time != null && (
                  <span className="text-xs px-2" style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}>
                    you · STT {fmt(msg.stt_time)}
                  </span>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-start gap-1">
                <div
                  className="rounded-2xl rounded-bl-sm px-4 py-3 w-full"
                  style={{ background: '#4361EE', border: '3px solid #000', boxShadow: '5px 5px 0px #000' }}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="font-bold text-sm" style={{ color: '#FFD60A' }}>Small O</span>
                    {msg.plugin && (
                      <span
                        className="text-xs px-2 py-0.5 rounded-lg font-bold"
                        style={{ background: '#FFD60A', color: '#000', border: '2px solid #000' }}
                      >
                        {PLUGIN_ICONS[msg.plugin] ?? '⚡'} {msg.plugin}
                      </span>
                    )}
                  </div>
                  <p className="text-base leading-relaxed" style={{ color: '#fff', fontWeight: 500 }}>
                    {msg.text}
                    {msg.streaming && (
                      <span
                        className="inline-block w-0.5 h-4 ml-0.5 align-middle animate-pulse"
                        style={{ background: '#FFD60A' }}
                      />
                    )}
                  </p>
                </div>
                {!msg.streaming && msg.llm_time != null && (
                  <span className="text-xs px-2" style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}>
                    ⚡ {fmt(msg.llm_time)}
                  </span>
                )}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>

      {messages.length === 0 && <EmptyState />}

      <div ref={bottomRef} />
    </div>
  )
}
