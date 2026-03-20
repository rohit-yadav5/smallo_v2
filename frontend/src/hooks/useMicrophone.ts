import { useCallback, useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'

const SILENCE_RMS   = 0.015   // RMS energy threshold — above = speech
const SILENCE_MS    = 1500    // ms of silence before we stop recording
const MIN_RECORD_MS = 400     // ignore clips shorter than this

export function useMicrophone() {
  const voiceState   = useAppStore((s) => s.voiceState)
  const wsConnected  = useAppStore((s) => s.wsConnected)
  const setMicActive = useAppStore((s) => s.setMicActive)

  // Stable ref so the AudioWorklet closure always reads the latest voice state
  // without needing to be re-created on every state change.
  const voiceStateRef = useRef(voiceState)
  useEffect(() => { voiceStateRef.current = voiceState }, [voiceState])

  const streamRef   = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const workletRef  = useRef<AudioWorkletNode | null>(null)
  const sourceRef   = useRef<MediaStreamAudioSourceNode | null>(null)
  const gainRef     = useRef<GainNode | null>(null)
  const activeRef   = useRef(false)

  // ── Teardown ─────────────────────────────────────────────────────────────
  // Always releases MediaStream tracks so the browser mic indicator goes away,
  // even when setup() failed before activeRef was set to true.
  const teardown = useCallback(() => {
    // Stop tracks unconditionally — this is what removes the red mic dot.
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null

    if (!activeRef.current) {
      // Partial setup — close any AudioContext that was created
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

  // ── Send audio to backend ─────────────────────────────────────────────────
  const sendAudio = useCallback((sampleRate: number, chunks: Float32Array[]) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('[mic] sendAudio: WS not open — audio dropped')
      return
    }
    const totalLen = chunks.reduce((a, c) => a + c.length, 0)
    const audio    = new Float32Array(totalLen)
    let offset = 0
    for (const c of chunks) { audio.set(c, offset); offset += c.length }

    // Protocol: [4-byte uint32 sampleRate] [Float32[] samples]
    const payload = new Uint8Array(4 + audio.byteLength)
    payload.set(new Uint8Array(new Uint32Array([sampleRate]).buffer), 0)
    payload.set(new Uint8Array(audio.buffer), 4)
    ws.send(payload.buffer)
    console.log(`[mic] ✓ sent  ${(totalLen / sampleRate).toFixed(2)}s  sr=${sampleRate}  ${(payload.byteLength / 1024).toFixed(1)} KB`)
  }, [])

  // ── Start mic ─────────────────────────────────────────────────────────────
  // MUST be called directly from a user-gesture handler (button click).
  // getUserMedia + AudioContext.resume() both require user activation.
  const startMic = useCallback(async () => {
    if (activeRef.current) return
    try {
      console.log('[mic] requesting getUserMedia...')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      streamRef.current = stream
      console.log('[mic] getUserMedia OK — track:', stream.getAudioTracks()[0]?.label)

      // AudioContext created right after the user-gesture chain — resumes correctly.
      const audioCtx = new AudioContext()
      audioCtxRef.current = audioCtx
      if (audioCtx.state === 'suspended') await audioCtx.resume()
      const actualSR = audioCtx.sampleRate
      console.log(`[mic] AudioContext state=${audioCtx.state}  sampleRate=${actualSR} Hz`)

      // Load processor from /public/smallo-pcm.js (served by Vite at root).
      await audioCtx.audioWorklet.addModule('/smallo-pcm.js')
      console.log('[mic] AudioWorklet loaded OK')

      const source     = audioCtx.createMediaStreamSource(stream)
      const worklet    = new AudioWorkletNode(audioCtx, 'smallo-pcm')
      const silentGain = audioCtx.createGain()
      silentGain.gain.value = 0          // don't play mic audio through speakers
      source.connect(worklet)
      worklet.connect(silentGain)
      silentGain.connect(audioCtx.destination)  // keeps AudioContext graph alive

      sourceRef.current  = source
      workletRef.current = worklet
      gainRef.current    = silentGain
      activeRef.current  = true

      // Re-resume AudioContext if the browser suspends it (e.g. tab loses focus).
      const keepAlive = () => {
        if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {})
      }
      document.addEventListener('click',     keepAlive)
      document.addEventListener('keydown',   keepAlive)
      document.addEventListener('touchstart', keepAlive)

      // ── Silence-based speech detection ────────────────────────────────────
      const chunks: Float32Array[] = []
      let isRecording      = false
      let recordingStartMs = 0
      let silenceStartMs   = 0
      let chunkCount       = 0

      worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
        const chunk = e.data
        chunkCount++
        if (chunkCount <= 3) console.log(`[mic] worklet chunk #${chunkCount}  len=${chunk.length}`)

        let sum = 0
        for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i]
        const rms = Math.sqrt(sum / chunk.length)

        const now   = Date.now()
        const state = voiceStateRef.current

        if (!isRecording) {
          if (state !== 'listening') return     // backend not ready, discard
          if (rms > SILENCE_RMS) {
            isRecording      = true
            recordingStartMs = now
            silenceStartMs   = now
            chunks.length    = 0
            chunks.push(chunk)
            console.log(`[mic] ▶ speech start  rms=${rms.toFixed(4)}`)
          }
        } else {
          chunks.push(chunk)
          if (rms > SILENCE_RMS) {
            silenceStartMs = now            // voice still active — reset silence clock
          } else {
            const recorded = now - recordingStartMs
            const silent   = now - silenceStartMs
            if (recorded >= MIN_RECORD_MS && silent >= SILENCE_MS) {
              console.log(`[mic] ■ speech end  ${(recorded / 1000).toFixed(2)}s`)
              sendAudio(actualSR, chunks)
              isRecording   = false
              chunks.length = 0
            }
          }
        }
      }

      setMicActive(true)
      console.log('[mic] ready — waiting for voiceState=listening')

    } catch (err) {
      console.error('[mic] FAILED:', err)
      // Always clean up every ref — prevents phantom mic indicator.
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current   = null
      audioCtxRef.current?.close().catch(() => {})
      audioCtxRef.current = null
      workletRef.current  = null
      gainRef.current     = null
      sourceRef.current   = null
      activeRef.current   = false
    }
  }, [setMicActive, sendAudio])

  // Release mic when backend disconnects (removes browser mic indicator)
  useEffect(() => {
    if (!wsConnected) teardown()
  }, [wsConnected, teardown])

  // Cleanup on unmount
  useEffect(() => () => teardown(), [teardown])

  return { startMic }
}
