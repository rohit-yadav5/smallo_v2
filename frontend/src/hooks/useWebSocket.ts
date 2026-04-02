import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'
import type { WSEvent } from '../types/events'

const WS_URL          = 'ws://localhost:8765'
const PING_INTERVAL   = 15_000   // send ping every 15 s
const PONG_TIMEOUT    = 5_000    // expect pong within 5 s
const MAX_RETRY_DELAY = 30_000   // cap exponential backoff at 30 s

export function useWebSocket() {
  const store            = useAppStore()
  const localWsRef       = useRef<WebSocket | null>(null)
  const llmStartRef      = useRef<number>(0)
  const currentMsgIdRef  = useRef<string | null>(null)
  const llmTimeRef       = useRef<number>(0)

  useEffect(() => {
    let retryDelay    = 1_000
    let reconnectTimer: ReturnType<typeof setTimeout>
    let pingInterval:  ReturnType<typeof setInterval>
    let pongTimeout:   ReturnType<typeof setTimeout>

    // ── Heartbeat helpers ──────────────────────────────────────────────────
    function startHeartbeat(ws: WebSocket) {
      clearInterval(pingInterval)
      clearTimeout(pongTimeout)
      pingInterval = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN) return
        ws.send(JSON.stringify({ event: 'ping' }))
        pongTimeout = setTimeout(() => {
          console.warn('[ws] pong timeout — forcing reconnect')
          ws.close()
        }, PONG_TIMEOUT)
      }, PING_INTERVAL)
    }

    function stopHeartbeat() {
      clearInterval(pingInterval)
      clearTimeout(pongTimeout)
    }

    // ── Finalize any in-progress streaming message ────────────────────────
    function finalizeStreaming() {
      if (currentMsgIdRef.current) {
        store.finalizeMessage(currentMsgIdRef.current, undefined)
        currentMsgIdRef.current = null
      }
    }

    // ── Connect ────────────────────────────────────────────────────────────
    function connect() {
      const ws = new WebSocket(WS_URL)
      localWsRef.current = ws
      wsRef.current      = ws

      // Safety: if neither onopen nor onclose fires within 10s, force-close
      const connectTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          console.warn('[ws] connection attempt timed out — closing')
          ws.close()
        }
      }, 10_000)

      ws.onopen = () => {
        clearTimeout(connectTimeout)
        console.log('[ws] connected')
        retryDelay = 1_000          // reset backoff on success
        store.setWsConnected(true)
        startHeartbeat(ws)
      }

      ws.onerror = (e) => {
        // onclose fires right after onerror — just log here.
        console.warn('[ws] error', e)
      }

      ws.onclose = () => {
        clearTimeout(connectTimeout)
        console.log(`[ws] closed — retry in ${retryDelay / 1000}s`)
        stopHeartbeat()
        wsRef.current = null
        store.setWsConnected(false)
        finalizeStreaming()           // fix: streaming bubble was left blinking forever
        reconnectTimer = setTimeout(connect, retryDelay)
        retryDelay = Math.min(retryDelay * 2, MAX_RETRY_DELAY)  // exponential backoff
      }

      ws.onmessage = (e) => {
        try {
          const msg: WSEvent = JSON.parse(e.data as string)
          handleEvent(msg)
        } catch (err) {
          console.warn('[ws] could not parse message:', err)
        }
      }
    }

    // ── Event handler ──────────────────────────────────────────────────────
    function handleEvent(msg: WSEvent) {
      switch (msg.event) {

        case 'VOICE_STATE':
          store.setVoiceState(msg.data.state)
          break

        case 'STT_RESULT':
          store.addUserMessage(
            msg.data.text,
            msg.data.recording_time,
            msg.data.transcription_time,
          )
          llmStartRef.current       = Date.now()
          currentMsgIdRef.current   = store.startAssistantMessage()
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
            store.finalizeMessage(
              currentMsgIdRef.current, 0, undefined,
              msg.data.plugin, msg.data.result,
            )
            currentMsgIdRef.current = null
          }
          break

        case 'MEMORY_EVENT':
          if (msg.data.retrieved) {
            store.glowMemoryNode(msg.data.id)
          } else {
            store.addMemoryNode({
              id:         msg.data.id,
              type:       msg.data.type,
              summary:    msg.data.summary,
              importance: msg.data.importance,
            })
          }
          break

        case 'SYSTEM_STATS':
          store.setSystemStats(msg.data)
          break

        case 'pong':
          clearTimeout(pongTimeout)   // received pong — cancel the close-on-timeout
          break
      }
    }

    connect()
    return () => {
      stopHeartbeat()
      clearTimeout(reconnectTimer)
      localWsRef.current?.close()
      wsRef.current = null
    }
  }, [])
}
