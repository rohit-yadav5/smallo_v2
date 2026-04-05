import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'
import type { WSEvent } from '../types/events'

// ── TTSAudioReceiver ──────────────────────────────────────────────────────────
// Receives chunked PCM16 or Opus audio frames from the server and plays them
// via Web Audio API in arrival order using a sequential drain queue.
//
// Protocol:
//   Server sends JSON: { event: "AUDIO_CHUNK", data: { format, sample_rate, chunk_count, chunk_ms } }
//   Server sends N binary ArrayBuffer frames (one per chunk)
//   Server sends JSON: { event: "AUDIO_DONE" }
//   On barge-in: Server sends JSON: { event: "AUDIO_ABORT" } → stop immediately

class TTSAudioReceiver {
  private _ctx:         AudioContext | null = null
  private _pending:     number  = 0    // binary frames still expected
  private _nextStart:   number  = 0    // AudioContext time to schedule next chunk
  private _sampleRate:  number  = 24000
  private _format:      string  = 'pcm16'
  private _nodes:       AudioBufferSourceNode[] = []

  private _getCtx(): AudioContext {
    if (!this._ctx || this._ctx.state === 'closed') {
      this._ctx     = new AudioContext({ sampleRate: this._sampleRate })
      this._nextStart = 0
    }
    return this._ctx
  }

  /** Called when the JSON metadata frame arrives. */
  onMetadata(format: string, sampleRate: number, chunkCount: number): void {
    this._format      = format
    this._sampleRate  = sampleRate
    this._pending     = chunkCount
    // Re-create context if sample rate changed
    if (this._ctx && this._ctx.sampleRate !== sampleRate) {
      this._ctx.close().catch(() => {})
      this._ctx = null
    }
    const ctx       = this._getCtx()
    this._nextStart = Math.max(this._nextStart, ctx.currentTime)
  }

  /** Called for each binary ArrayBuffer frame. */
  onBinaryFrame(buffer: ArrayBuffer): void {
    if (this._pending <= 0) return
    this._pending--

    const ctx      = this._getCtx()
    const float32  = this._decode(buffer)
    if (float32.length === 0) return

    const audioBuf = ctx.createBuffer(1, float32.length, this._sampleRate)
    audioBuf.copyToChannel(float32, 0)

    const src = ctx.createBufferSource()
    src.buffer = audioBuf
    src.connect(ctx.destination)
    src.start(this._nextStart)
    this._nextStart += audioBuf.duration
    this._nodes.push(src)

    src.onended = () => {
      this._nodes = this._nodes.filter(n => n !== src)
      src.disconnect()
    }
  }

  /** Called when AUDIO_DONE arrives — nodes drain via onended; log remaining count. */
  onDone(): void {
    this._pending = 0
    console.debug(`[tts] AUDIO_DONE — ${this._nodes.length} nodes still scheduled`)
  }

  /** Called when AUDIO_ABORT arrives — stop all queued audio immediately. */
  stop(): void {
    this._pending   = 0
    this._nextStart = 0
    for (const node of this._nodes) {
      try { node.stop() } catch { /* already stopped */ }
      node.disconnect()
    }
    this._nodes = []
  }

  private _decode(buffer: ArrayBuffer): Float32Array {
    if (this._format === 'pcm16') {
      const int16  = new Int16Array(buffer)
      const float  = new Float32Array(int16.length)
      for (let i = 0; i < int16.length; i++) {
        float[i] = int16[i] / 32768
      }
      return float
    }
    // Opus: not decodable natively — log and skip
    console.warn('[tts] opus format not supported in browser; use pcm16')
    return new Float32Array(0)
  }
}

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
  const audioReceiverRef = useRef<TTSAudioReceiver>(new TTSAudioReceiver())

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

      ws.binaryType = 'arraybuffer'

      ws.onmessage = (e) => {
        // Binary frame: raw audio chunk from TTS
        if (e.data instanceof ArrayBuffer) {
          audioReceiverRef.current.onBinaryFrame(e.data)
          return
        }
        // JSON frame
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

        case 'STT_PARTIAL':
          store.setPartialUserText(msg.data.text, msg.data.hypothesis ?? '')
          break

        case 'STT_RESULT':
          store.clearPartialUserText()
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

        case 'AUDIO_CHUNK':
          audioReceiverRef.current.onMetadata(
            msg.data.format,
            msg.data.sample_rate,
            msg.data.chunk_count,
          )
          break

        case 'AUDIO_DONE':
          audioReceiverRef.current.onDone()
          break

        case 'AUDIO_ABORT':
          audioReceiverRef.current.stop()
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
