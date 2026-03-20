import { useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'

const SILENCE_RMS   = 0.015
const SILENCE_MS    = 1500
const MIN_RECORD_MS = 400

export function useMicrophone() {
  const voiceState    = useAppStore((s) => s.voiceState)
  const wsConnected   = useAppStore((s) => s.wsConnected)
  const voiceStateRef = useRef(voiceState)

  // Refs for cleanup/restart across effects
  const streamRef      = useRef<MediaStream | null>(null)
  const audioCtxRef    = useRef<AudioContext | null>(null)
  const workletRef     = useRef<AudioWorkletNode | null>(null)
  const sourceRef      = useRef<MediaStreamAudioSourceNode | null>(null)
  const gainRef        = useRef<GainNode | null>(null)
  const activeRef      = useRef(false)   // true while audio pipeline is running

  useEffect(() => {
    voiceStateRef.current = voiceState
  }, [voiceState])

  // ── Stop mic completely (releases browser mic indicator) ─────────────
  function teardown() {
    if (!activeRef.current) return
    activeRef.current = false
    workletRef.current?.port.close()
    workletRef.current?.disconnect()
    gainRef.current?.disconnect()
    sourceRef.current?.disconnect()
    streamRef.current?.getTracks().forEach((t) => t.stop())   // releases 🔴 mic dot
    audioCtxRef.current?.close()
    workletRef.current = null
    gainRef.current    = null
    sourceRef.current  = null
    streamRef.current  = null
    audioCtxRef.current= null
    console.log('[mic] teardown — mic released')
  }

  // ── Start mic ─────────────────────────────────────────────────────────
  async function setup() {
    if (activeRef.current) return
    try {
      console.log('[mic] requesting getUserMedia...')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      streamRef.current = stream
      console.log('[mic] getUserMedia OK — track:', stream.getAudioTracks()[0]?.label)

      const audioCtx = new AudioContext()
      audioCtxRef.current = audioCtx
      if (audioCtx.state === 'suspended') await audioCtx.resume()
      const actualSR = audioCtx.sampleRate
      console.log(`[mic] AudioContext  state=${audioCtx.state}  sampleRate=${actualSR} Hz`)

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
          if (state !== 'listening') return
          if (rms > SILENCE_RMS) {
            isRecording = true; recordingStartMs = now; silenceStartMs = now
            chunks.length = 0; chunks.push(chunk)
            console.log(`[mic] ▶ speech start  rms=${rms.toFixed(4)}`)
          }
        } else {
          chunks.push(chunk)
          if (rms > SILENCE_RMS) {
            silenceStartMs = now
          } else {
            const recorded = now - recordingStartMs
            const silent   = now - silenceStartMs
            if (recorded >= MIN_RECORD_MS && silent >= SILENCE_MS) {
              console.log(`[mic] ■ speech end  ${(recorded/1000).toFixed(2)}s`)
              sendAudio(actualSR, chunks)
              isRecording = false; chunks.length = 0
            }
          }
        }
      }

      console.log('[mic] ready — waiting for voiceState=listening')
    } catch (err) {
      console.error('[mic] FAILED:', err)
      activeRef.current = false
    }
  }

  function sendAudio(sampleRate: number, chunks: Float32Array[]) {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('[mic] sendAudio: WS not ready — dropped'); return
    }
    const totalLen = chunks.reduce((a, c) => a + c.length, 0)
    const audio    = new Float32Array(totalLen)
    let offset = 0
    for (const c of chunks) { audio.set(c, offset); offset += c.length }
    const payload = new Uint8Array(4 + audio.byteLength)
    payload.set(new Uint8Array(new Uint32Array([sampleRate]).buffer), 0)
    payload.set(new Uint8Array(audio.buffer), 4)
    ws.send(payload.buffer)
    console.log(`[mic] ✓ sent  ${(totalLen/sampleRate).toFixed(2)}s  sr=${sampleRate}  ${(payload.byteLength/1024).toFixed(1)} KB`)
  }

  // ── React to connection changes ───────────────────────────────────────
  useEffect(() => {
    if (wsConnected) {
      setup()
    } else {
      teardown()
    }
  }, [wsConnected])

  // ── Cleanup on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => teardown()
  }, [])
}
