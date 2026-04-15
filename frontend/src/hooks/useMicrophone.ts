import { useCallback, useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'

export function useMicrophone() {
  const voiceState   = useAppStore((s) => s.voiceState)
  const wsConnected  = useAppStore((s) => s.wsConnected)
  const setMicActive = useAppStore((s) => s.setMicActive)
  const sttEnabled   = useAppStore((s) => s.sttEnabled)

  // Stable ref so the AudioWorklet message handler always reads the latest value
  // without needing to be recreated.
  const sttEnabledRef = useRef(sttEnabled)
  useEffect(() => { sttEnabledRef.current = sttEnabled }, [sttEnabled])

  // Stable ref — AudioWorklet closure always reads latest voice state
  const voiceStateRef = useRef(voiceState)
  useEffect(() => { voiceStateRef.current = voiceState }, [voiceState])

  const streamRef    = useRef<MediaStream | null>(null)
  const audioCtxRef  = useRef<AudioContext | null>(null)
  const workletRef   = useRef<AudioWorkletNode | null>(null)
  const sourceRef    = useRef<MediaStreamAudioSourceNode | null>(null)
  const gainRef      = useRef<GainNode | null>(null)
  const activeRef    = useRef(false)
  // Store keepAlive ref so we can remove it in teardown (fixes listener leak)
  const keepAliveRef = useRef<(() => void) | null>(null)

  // ── Teardown ───────────────────────────────────────────────────────────
  const teardown = useCallback(() => {
    // Remove keepAlive listeners before anything else
    if (keepAliveRef.current) {
      document.removeEventListener('click',      keepAliveRef.current)
      document.removeEventListener('keydown',    keepAliveRef.current)
      document.removeEventListener('touchstart', keepAliveRef.current)
      keepAliveRef.current = null
    }

    // Always stop tracks — removes browser mic indicator
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null

    if (!activeRef.current) {
      audioCtxRef.current?.close().catch(() => {})
      audioCtxRef.current = null
      return
    }

    activeRef.current = false
    workletRef.current?.port.close()
    workletRef.current?.disconnect()
    gainRef.current?.disconnect()
    sourceRef.current?.disconnect()
    audioCtxRef.current?.close().catch(() => {})
    workletRef.current  = null
    gainRef.current     = null
    sourceRef.current   = null
    audioCtxRef.current = null
    setMicActive(false)
    console.log('[mic] teardown — mic released')
  }, [setMicActive])

  // ── Send a single audio chunk to backend ──────────────────────────────
  // Protocol: [4-byte uint32 sampleRate][Float32[] samples]
  const sendChunk = useCallback((sampleRate: number, chunk: Float32Array) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    const header  = new Uint32Array([sampleRate])
    const payload = new Uint8Array(4 + chunk.byteLength)
    payload.set(new Uint8Array(header.buffer), 0)
    // Use byteOffset/byteLength in case chunk is a view into a larger buffer
    payload.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), 4)
    ws.send(payload.buffer)
  }, [])

  // ── Start mic ──────────────────────────────────────────────────────────
  // MUST be called from a user-gesture handler (button click).
  const startMic = useCallback(async () => {
    if (activeRef.current) return
    try {
      console.log('[mic] requesting getUserMedia...')
      // Explicit AEC constraints — without these the browser may not apply
      // acoustic echo cancellation, causing the bot's own TTS output (played
      // through speakers) to be picked up by the mic and trigger false barge-ins.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation:  true,
          noiseSuppression:  true,
          autoGainControl:   true,
          channelCount:      1,
        },
        video: false,
      })
      streamRef.current = stream
      console.log('[mic] getUserMedia OK — track:', stream.getAudioTracks()[0]?.label)

      const audioCtx = new AudioContext()
      audioCtxRef.current = audioCtx
      if (audioCtx.state === 'suspended') await audioCtx.resume()
      const actualSR = audioCtx.sampleRate
      console.log(`[mic] AudioContext state=${audioCtx.state}  sampleRate=${actualSR} Hz`)

      await audioCtx.audioWorklet.addModule('/smallo-pcm.js')
      console.log('[mic] AudioWorklet loaded OK')

      const source     = audioCtx.createMediaStreamSource(stream)
      const worklet    = new AudioWorkletNode(audioCtx, 'smallo-pcm')
      const silentGain = audioCtx.createGain()
      silentGain.gain.value = 0
      source.connect(worklet)
      worklet.connect(silentGain)
      silentGain.connect(audioCtx.destination)

      sourceRef.current  = source
      workletRef.current = worklet
      gainRef.current    = silentGain
      activeRef.current  = true

      // Re-resume AudioContext when tab regains focus
      const keepAlive = () => {
        if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {})
      }
      keepAliveRef.current = keepAlive
      document.addEventListener('click',      keepAlive)
      document.addEventListener('keydown',    keepAlive)
      document.addEventListener('touchstart', keepAlive)

      let chunkCount = 0

      worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
        const chunk = e.data
        chunkCount++
        if (chunkCount <= 3) {
          console.log(`[mic] worklet chunk #${chunkCount}  len=${chunk.length}  sr=${actualSR}`)
        }

        const state = voiceStateRef.current

        // Gate 1: STT disabled — drop frame entirely, no data to backend.
        if (!sttEnabledRef.current) return

        // Gate 2: Send during listening (VAD captures speech) and speaking (VAD detects barge-in).
        // Discard during idle/thinking to avoid queue buildup.
        if (state === 'listening' || state === 'speaking') {
          sendChunk(actualSR, chunk)
        }
      }

      setMicActive(true)
      console.log('[mic] ready — streaming to backend VAD')

    } catch (err) {
      console.error('[mic] FAILED:', err)
      // Clean up everything on error to prevent phantom mic indicator
      if (keepAliveRef.current) {
        document.removeEventListener('click',      keepAliveRef.current)
        document.removeEventListener('keydown',    keepAliveRef.current)
        document.removeEventListener('touchstart', keepAliveRef.current)
        keepAliveRef.current = null
      }
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current   = null
      audioCtxRef.current?.close().catch(() => {})
      audioCtxRef.current = null
      workletRef.current  = null
      gainRef.current     = null
      sourceRef.current   = null
      activeRef.current   = false
    }
  }, [setMicActive, sendChunk])

  // Release mic when backend disconnects
  useEffect(() => {
    if (!wsConnected) teardown()
  }, [wsConnected, teardown])

  // ── STT toggle: start/stop mic when sttEnabled changes ────────────────
  // Tracks whether mic was running before being disabled so we can auto-restart.
  const wasActiveBeforeDisableRef = useRef(false)
  useEffect(() => {
    if (!sttEnabled) {
      // Remember if mic was running so we can restore when re-enabled
      wasActiveBeforeDisableRef.current = activeRef.current
      teardown()
      console.log('[mic] STT disabled — worklet torn down')
    } else if (wasActiveBeforeDisableRef.current && wsConnected) {
      // Re-enable: auto-restart if the mic was running before.
      // This effect fires synchronously after the toggle click (user gesture),
      // so getUserMedia permission is available in most browsers.
      wasActiveBeforeDisableRef.current = false
      console.log('[mic] STT re-enabled — restarting mic')
      startMic()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sttEnabled])

  // Cleanup on unmount
  useEffect(() => () => teardown(), [teardown])

  return { startMic }
}
