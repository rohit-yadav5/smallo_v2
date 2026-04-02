"""
vad/streaming.py — Stateful streaming VAD built on SileroEngine + RingBuffer.

Architecture
────────────
Inspired by vad_2/vad_engine_silero and vad_2/buffer/silero_buffer.

Audio flow
──────────
raw chunks (any size, any SR)
    ↓  add_audio()           ← called by the WebSocket ingest thread
  RingBuffer  (3 s @ 16 kHz)
    ↓  process()             ← called by the VAD processing thread every ~20 ms
  SileroEngine.process_frame()  (512-sample windows, step=256 for 50% overlap)
    ↓  state machine
  complete utterance → _pipeline_loop  (via _speech_queue)

Barge-in safety
───────────────
A 400 ms grace period is enforced by _vad_loop in backend/main.py after
transitioning to "speaking" state.  This prevents the bot's own TTS audio
(echo through the microphone) from triggering a false barge-in.

Additionally, process_frame() requires BARGE_IN_MIN_CONSECUTIVE (2) consecutive
speech frames for a barge-in trigger, avoiding single-frame noise bursts.

Public API
──────────
vad = StreamingVAD()
utterance = vad.process(chunk_16k)   # feed audio; returns np.ndarray or None
vad.reset()                          # reset LSTM + all state (between turns)
vad.is_speaking                      # True while accumulating speech
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vad.engine import SileroEngine
from vad.ring_buffer import RingBuffer

_SR     = 16_000
_WINDOW = 512    # Silero requirement
_STEP   = 256    # 50 % overlap → 16 ms step → smoother probability estimates


class StreamingVAD:
    """
    Stateful streaming wrapper.

    Feed 16 kHz float32 audio via process(); get complete utterances back.

    Parameters
    ──────────
    onset_threshold   — Silero probability to start speech accumulation (0–1)
    offset_threshold  — Silero probability below which silence is counted (0–1)
    silence_ms        — consecutive silence (ms) before sealing an utterance
    min_speech_ms     — minimum utterance duration to keep (rejects noise blips)
    max_speech_s      — force-flush after this many seconds (safety net)
    pre_pad_ms        — pre-speech audio prepended (captures word beginnings)
    onset_count       — consecutive speech windows needed before onset is declared
                        (higher = fewer false positives from brief noise)
    """

    def __init__(
        self,
        onset_threshold:  float = 0.50,
        offset_threshold: float = 0.35,
        silence_ms:       int   = 600,
        min_speech_ms:    int   = 120,   # 120 ms: captures short words (yes/no/ok)
        max_speech_s:     int   = 30,
        pre_pad_ms:       int   = 200,
        onset_count:      int   = 2,
        first_silence_cb  = None,        # callable(audio_snapshot) — called once at
                                         # the first silence frame so the caller can
                                         # start STT in the background while the
                                         # silence confirmation window (silence_ms)
                                         # elapses.  Called at most once per utterance.
    ):
        self._engine = SileroEngine(threshold=onset_threshold)
        self._offset = offset_threshold

        # Pre-speech ring buffer — holds audio from just before speech onset
        # so the first word (which starts before prob >= threshold) is captured.
        pre_s = pre_pad_ms / 1000 + 0.1       # a little extra margin
        self._pre = RingBuffer(max_seconds=pre_s)

        # Speech accumulation (sequential; added in _STEP increments)
        self._speech:    list[np.ndarray] = []
        self._n_samples: int = 0

        # Thresholds in samples / window counts
        self._silence_threshold = max(1, int(_SR * silence_ms   / 1000 / _STEP))
        self._min_samples       = int(_SR * min_speech_ms / 1000)
        self._max_samples       = int(_SR * max_speech_s)
        self._pre_samples       = int(_SR * pre_pad_ms    / 1000)

        # Onset hysteresis — require consecutive speech windows before onset
        self._onset_count    = onset_count
        self._onset_buf: int = 0   # consecutive-above-threshold counter

        # Early-STT callback — fired once per utterance at first silence frame
        self._first_silence_cb   = first_silence_cb
        self._early_stt_fired    = False  # guard: fire at most once per utterance

        # Runtime state
        self._speaking:  bool = False
        self._silence:   int  = 0   # consecutive sub-offset step count
        self._leftover:  np.ndarray = np.empty(0, dtype=np.float32)

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, audio_16k: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed a chunk of 16 kHz float32 audio.

        Returns a complete utterance (float32 @ 16 kHz) when speech ends,
        or None while still accumulating / waiting for speech.
        """
        # Stitch leftover from previous call
        if self._leftover.size:
            audio_16k = np.concatenate([self._leftover, audio_16k])

        result: Optional[np.ndarray] = None
        pos = 0

        # Process in _STEP increments (50 % overlap: window=512, step=256)
        while pos + _WINDOW <= len(audio_16k):
            window = audio_16k[pos : pos + _WINDOW]
            prob, is_speech_raw = self._engine.process_frame(window)

            if self._speaking:
                # Add STEP new samples to the speech buffer (avoid overlap double-count)
                new_chunk = audio_16k[pos : pos + _STEP]
                self._speech.append(new_chunk)
                self._n_samples += len(new_chunk)

                if prob < self._offset:
                    self._silence += 1
                    # First silence frame: fire early-STT callback so the caller
                    # can start Whisper in the background while the silence
                    # confirmation window (silence_ms) elapses.  Guarded by
                    # _early_stt_fired so mid-speech pauses don't re-trigger it.
                    if self._silence == 1 and not self._early_stt_fired \
                            and self._first_silence_cb is not None:
                        self._early_stt_fired = True
                        snapshot = (np.concatenate(self._speech)
                                    if self._speech else np.empty(0, dtype=np.float32))
                        try:
                            self._first_silence_cb(snapshot)
                        except Exception:
                            pass
                    if self._silence >= self._silence_threshold:
                        result = self._flush()
                else:
                    self._silence = 0

                # Force-flush if utterance is too long
                if self._n_samples >= self._max_samples:
                    result = self._flush(forced=True)

            else:
                # Onset hysteresis: require consecutive speech windows before
                # committing to speech.  Check BEFORE adding to _pre so that
                # the onset step is never written into the ring and then read
                # back out again — that would place it twice in _speech
                # (once in pre_audio, once as _speech[1]), creating a 16 ms
                # duplicate phoneme at the very start of every utterance.
                # Symptom: Whisper mis-decodes the first word on every turn
                # ("what is your name" → "i is your name", etc.).
                if is_speech_raw:
                    self._onset_buf += 1
                else:
                    self._onset_buf = max(0, self._onset_buf - 1)

                if self._onset_buf >= self._onset_count:
                    # Onset fires.  Current step is the first real speech chunk.
                    # _pre does NOT contain it yet (we skipped the add above),
                    # so pre_audio + current_step has no overlap.
                    self._onset_buf = 0
                    self._speaking  = True
                    self._silence   = 0
                    pre_audio = self._pre.get_last_samples(self._pre_samples)
                    self._speech    = [pre_audio, audio_16k[pos : pos + _STEP]]
                    self._n_samples = len(pre_audio) + _STEP
                    print(f"  [vad] ▶  speech start  prob={prob:.3f}", flush=True)
                else:
                    # Not onset yet — keep ring warm for word-start padding.
                    self._pre.add_frames(audio_16k[pos : pos + _STEP])

            pos += _STEP

        # Keep remainder for next call
        self._leftover = audio_16k[pos:].copy()
        return result

    def reset(self) -> None:
        """
        Full reset — clears all buffers AND Silero LSTM hidden states.
        Call between pipeline turns for a clean detection state.

        _pre ring is now also cleared so that TTS echo accumulated during the
        barge-in grace period is not prepended to the next utterance and fed to
        Whisper.  The ring re-fills naturally within pre_pad_ms of new audio.
        """
        self._engine.reset_states()
        self._speech         = []
        self._n_samples      = 0
        self._silence        = 0
        self._speaking       = False
        self._onset_buf      = 0
        self._early_stt_fired = False
        self._leftover       = np.empty(0, dtype=np.float32)
        self._pre.clear()   # ← wipe echo / stale audio from previous state

    @property
    def is_speaking(self) -> bool:
        """True while a speech segment is being accumulated (not yet sealed)."""
        return self._speaking

    # ── Private ────────────────────────────────────────────────────────────

    def _flush(self, forced: bool = False) -> np.ndarray:
        """Seal the current speech segment and return the audio."""
        audio = (np.concatenate(self._speech)
                 if self._speech else np.empty(0, dtype=np.float32))
        dur = len(audio) / _SR

        if self._n_samples >= self._min_samples:
            tag = "forced" if forced else "end"
            print(f"  [vad] ■  speech {tag}  {dur:.2f}s", flush=True)
        else:
            print(f"  [vad] ✗  too short ({dur:.3f}s < min {self._min_samples/_SR:.3f}s) — discarding",
                  flush=True)
            audio = np.empty(0, dtype=np.float32)

        # Reset per-utterance state (LSTM hidden states preserved until reset() called)
        self._speech          = []
        self._n_samples       = 0
        self._silence         = 0
        self._speaking        = False
        self._early_stt_fired = False
        return audio
