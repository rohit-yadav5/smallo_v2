import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import type { WSEvent } from '../types/events'

const WS_URL = 'ws://localhost:8765'

export function useWebSocket() {
  const store = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const llmStartRef = useRef<number>(0)
  const currentMsgIdRef = useRef<string | null>(null)
  const llmTimeRef = useRef<number>(0)

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>

    function connect() {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onmessage = (e) => {
        try {
          const msg: WSEvent = JSON.parse(e.data)
          handleEvent(msg)
        } catch {}
      }

      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    function handleEvent(msg: WSEvent) {
      switch (msg.event) {
        case 'VOICE_STATE':
          store.setVoiceState(msg.data.state)
          break

        case 'STT_RESULT':
          store.addUserMessage(msg.data.text, msg.data.recording_time, msg.data.transcription_time)
          llmStartRef.current = Date.now()
          currentMsgIdRef.current = store.startAssistantMessage()
          break

        case 'LLM_TOKEN':
          if (currentMsgIdRef.current) {
            if (!msg.data.done) {
              store.appendToken(currentMsgIdRef.current, msg.data.token)
            } else {
              llmTimeRef.current = (Date.now() - llmStartRef.current) / 1000
              store.finalizeMessage(currentMsgIdRef.current, llmTimeRef.current)
              currentMsgIdRef.current = null
            }
          }
          break

        case 'PLUGIN_ACTION':
          store.addPluginNotification({
            plugin: msg.data.plugin,
            action: msg.data.action,
            result: msg.data.result,
          })
          if (msg.data.direct && currentMsgIdRef.current) {
            // Direct plugin: no LLM tokens will follow — populate text and finalize now
            store.finalizeMessage(currentMsgIdRef.current, 0, undefined, msg.data.plugin, msg.data.result)
            currentMsgIdRef.current = null
          }
          // Summarized plugin: leave currentMsgIdRef intact so LLM_TOKEN events fill the message
          break

        case 'MEMORY_EVENT':
          if (msg.data.retrieved) {
            store.glowMemoryNode(msg.data.id)
          } else {
            store.addMemoryNode({
              id: msg.data.id,
              type: msg.data.type,
              summary: msg.data.summary,
              importance: msg.data.importance,
            })
          }
          break

        case 'SYSTEM_STATS':
          store.setSystemStats(msg.data)
          break
      }
    }

    connect()
    return () => {
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [])
}
