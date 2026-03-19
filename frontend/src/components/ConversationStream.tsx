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

export function ConversationStream() {
  const messages = useAppStore((s) => s.messages)
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
              /* ── User bubble — sunny yellow ── */
              <div className="flex flex-col items-end gap-1">
                <div
                  className="rounded-2xl rounded-br-sm px-4 py-3 max-w-[85%]"
                  style={{
                    background: '#FFD60A',
                    border: '3px solid #000000',
                    boxShadow: '5px 5px 0px #000000',
                  }}
                >
                  <p className="text-base leading-relaxed" style={{ color: '#000000', fontWeight: 600 }}>
                    {msg.text}
                  </p>
                </div>
                {msg.stt_time != null && (
                  <span
                    className="text-xs px-2"
                    style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}
                  >
                    you · STT {fmt(msg.stt_time)}
                  </span>
                )}
              </div>
            ) : (
              /* ── AI bubble — mint green ── */
              <div className="flex flex-col items-start gap-1">
                <div
                  className="rounded-2xl rounded-bl-sm px-4 py-3 w-full"
                  style={{
                    background: '#4361EE',
                    border: '3px solid #000000',
                    boxShadow: '5px 5px 0px #000000',
                  }}
                >
                  {/* Header */}
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="font-bold text-sm" style={{ color: '#FFD60A' }}>
                      Small O
                    </span>
                    {msg.plugin && (
                      <span
                        className="text-xs px-2 py-0.5 rounded-lg font-bold"
                        style={{ background: '#FFD60A', color: '#000000', border: '2px solid #000000' }}
                      >
                        {PLUGIN_ICONS[msg.plugin] ?? '⚡'} {msg.plugin}
                      </span>
                    )}
                  </div>

                  <p className="text-base leading-relaxed" style={{ color: '#ffffff', fontWeight: 500 }}>
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
                  <span
                    className="text-xs px-2"
                    style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(0,0,0,0.4)' }}
                  >
                    ⚡ {fmt(msg.llm_time)}
                  </span>
                )}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>

      {messages.length === 0 && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-4xl mb-3">🎤</p>
            <p className="text-base font-semibold" style={{ color: 'rgba(0,0,0,0.45)' }}>
              Say something to get started!
            </p>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
