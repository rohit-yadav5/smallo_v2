import { useEffect, useRef } from 'react'
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

// ── Empty-state shown when there are no messages yet ──────────────────────
function EmptyState({ onStartMic }: { onStartMic: () => Promise<void> }) {
  const wsConnected = useAppStore((s) => s.wsConnected)
  const voiceState  = useAppStore((s) => s.voiceState)
  const micActive   = useAppStore((s) => s.micActive)

  // ── Not connected ──────────────────────────────────────────────────────
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
          <div className="flex gap-1.5 mt-1">
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

  // ── Connected but models still loading (voiceState = idle) ────────────
  if (voiceState === 'idle') {
    return (
      <motion.div
        className="flex-1 flex items-center justify-center"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      >
        <div className="text-center flex flex-col items-center gap-4">
          <motion.div
            className="w-20 h-20 rounded-2xl flex items-center justify-center text-4xl"
            style={{ background: '#3A0CA3', border: '4px solid #000', boxShadow: '6px 6px 0px #000' }}
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          >
            🤖
          </motion.div>
          <div
            className="px-4 py-1.5 rounded-xl"
            style={{ background: '#3A0CA3', border: '3px solid #000', boxShadow: '4px 4px 0px #000' }}
          >
            <p className="text-sm font-black tracking-widest uppercase" style={{ color: '#FFD60A' }}>
              Loading models...
            </p>
          </div>
          <div className="flex gap-1.5">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: '#7B2FBE', border: '2px solid #000' }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1.0, repeat: Infinity, delay: i * 0.33 }}
              />
            ))}
          </div>
        </div>
      </motion.div>
    )
  }

  // ── Ready but mic not yet started — show Tap to Start ─────────────────
  if (!micActive) {
    return (
      <motion.div
        className="flex-1 flex items-center justify-center"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 22 }}
      >
        <div className="text-center flex flex-col items-center gap-4">
          <motion.button
            onClick={onStartMic}
            className="w-28 h-28 rounded-3xl flex items-center justify-center text-5xl cursor-pointer"
            style={{
              background:  '#4CC9F0',
              border:      '5px solid #000',
              boxShadow:   '8px 8px 0px #000',
            }}
            whileHover={{ scale: 1.05, boxShadow: '10px 10px 0px #000' }}
            whileTap={{   scale: 0.95, boxShadow: '4px 4px 0px #000', y: 4 }}
          >
            🎤
          </motion.button>
          <div
            className="px-5 py-2 rounded-xl"
            style={{ background: '#4CC9F0', border: '3px solid #000', boxShadow: '5px 5px 0px #000' }}
          >
            <p className="text-base font-black" style={{ color: '#000' }}>Tap to Start</p>
          </div>
        </div>
      </motion.div>
    )
  }

  // ── Mic active — ready to speak ────────────────────────────────────────
  return (
    <motion.div
      className="flex-1 flex items-center justify-center"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
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

// ── Main export ────────────────────────────────────────────────────────────
export function ConversationStream({ onStartMic }: { onStartMic: () => Promise<void> }) {
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
                  <span className="text-xs px-2"
                    style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}>
                    you · {fmt(msg.stt_time)}
                    {msg.transcription_time != null && ` · STT ${fmt(msg.transcription_time)}`}
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
                  <span className="text-xs px-2"
                    style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}>
                    ⚡ {fmt(msg.llm_time)}
                  </span>
                )}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>

      {messages.length === 0 && <EmptyState onStartMic={onStartMic} />}

      <div ref={bottomRef} />
    </div>
  )
}
