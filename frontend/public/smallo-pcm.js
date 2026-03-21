/**
 * smallo-pcm.js  — AudioWorklet processor
 *
 * Runs in the AUDIO RENDERING THREAD (not the JS main thread).
 * This means it fires reliably even when the browser tab is in the background
 * or the user has switched to another window.
 *
 * It accumulates 128-sample engine callbacks into larger chunks (4096 samples)
 * before posting to the main thread, to reduce message overhead.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._buf       = []
    this._size      = 0
    this._chunkSize = 4096   // post to main thread every 4096 samples (~256 ms @ 16 kHz)
  }

  process(inputs) {
    const ch = inputs[0]?.[0]
    if (!ch || ch.length === 0) return true

    // Copy the channel data — the engine reuses the buffer after this call
    this._buf.push(new Float32Array(ch))
    this._size += ch.length

    if (this._size >= this._chunkSize) {
      // Flatten into one contiguous array and transfer ownership (zero-copy)
      const out = new Float32Array(this._size)
      let off = 0
      for (const b of this._buf) { out.set(b, off); off += b.length }
      this.port.postMessage(out, [out.buffer])
      this._buf  = []
      this._size = 0
    }

    return true   // returning false would destroy the processor
  }
}

registerProcessor('smallo-pcm', PCMProcessor)
