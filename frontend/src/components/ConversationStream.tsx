import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

const PLUGIN_ICONS: Record<string, string> = {
  web: '🌐',
  computer: '💻',
  security: '🔒',
}

export function ConversationStream() {
  const messages = useAppStore((s) => s.messages)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col gap-4 overflow-y-auto h-full px-2 py-2 scrollbar-thin">
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="flex flex-col gap-1"
          >
            <div className="flex items-start gap-2">
              <span
                className="text-xs mt-0.5 shrink-0"
                style={{
                  fontFamily: 'Orbitron, sans-serif',
                  color: msg.role === 'user' ? '#a1a1aa' : '#8b5cf6',
                }}
              >
                {msg.role === 'user' ? 'YOU' : 'SMALL O'}
              </span>
              <p
                className="text-sm leading-relaxed"
                style={{
                  fontFamily: 'Inter, sans-serif',
                  color: msg.role === 'user' ? '#d4d4d8' : '#e4e4e7',
                }}
              >
                {msg.text}
                {msg.streaming && (
                  <span className="inline-block w-1 h-3.5 ml-0.5 bg-purple-400 animate-pulse align-middle" />
                )}
              </p>
            </div>

            {!msg.streaming && msg.role === 'assistant' && (
              <div
                className="flex items-center gap-3 ml-10 text-xs"
                style={{ fontFamily: 'JetBrains Mono, monospace', color: '#52525b' }}
              >
                {msg.stt_time && <span>STT {msg.stt_time.toFixed(2)}s</span>}
                {msg.llm_time && <span>LLM {msg.llm_time.toFixed(2)}s</span>}
                {msg.tts_time && <span>TTS {msg.tts_time.toFixed(2)}s</span>}
                {msg.plugin && (
                  <span className="text-purple-500">
                    {PLUGIN_ICONS[msg.plugin] || '⚡'} {msg.plugin}
                  </span>
                )}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  )
}
