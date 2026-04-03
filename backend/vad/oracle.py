"""
vad/oracle.py — VAD as a timestamp oracle.

Architecture
────────────
VADOracle watches the audio stream and fires callbacks when speech starts
and ends.  It does NOT accumulate audio, does NOT return utterances, and
does NOT gate the pipeline.  Its only job is to say WHEN speech happened.

  Mic → RollingAudioBuffer (always)
    └──► VADOracle.process() ─► on_speech_start(start_s)
                              ─► on_speech_end(start_s, end_s)

The caller (main.py _audio_ingestion_loop) passes these timestamps to
RollingAudioBuffer.read_window() to extract the actual audio.

Why this is better than StreamingVAD
─────────────────────────────────────
StreamingVAD accumulated audio into _speech and _pre lists.  When the
speaking→listening state transition called vad.reset(), those lists were
wiped, destroying the user's first word.

VADOracle never holds audio, so there is nothing to lose on reset.
The rolling buffer is the single source of truth for audio data.

Pre/post buffer
───────────────
pre_buffer_s  is SUBTRACTED from the raw onset timestamp.  The extracted
window therefore starts pre_buffer_s before VAD fired — capturing the
complete first word even if Silero was late to detect it.

post_buffer_s is ADDED to the raw offset timestamp.  This ensures the
last word isn't clipped when silence is detected slightly early.

Both are configurable (default 2.0 s each, per user's spec).

Callbacks
─────────
on_speech_start(start_s)          — confirmed onset (pre_buffer already applied)
on_speech_end(start_s, end_s)     — post_buffer already applied to end_s
on_speech_chunk(chunk: ndarray)   — every 16 ms speech frame (live display)
on_first_silence(snapshot: ndarray) — first silence frame (early-STT trigger)
"""
from __future__ import annotations

import numpy as np

from vad.engine import SileroEngine

_SR     = 16_000
_WINDOW = 512    # Silero requirement (32 ms)
_STEP   = 256    # 50 % overlap → 16 ms per step


class VADOracle:
    """
    Timestamp-only VAD wrapper around SileroEngine.

    Parameters
    ──────────
    onset_threshold   — Silero probability to begin onset counting (0–1)
    offset_threshold  — probability below which silence is counted (0–1)
    onset_count       — consecutive speech frames to confirm speech start
                        (hysteresis; same role as onset_count in StreamingVAD)
    offset_count      — consecutive silence frames to confirm speech end
                        45 frames × 16 ms/frame = 720 ms silence window
    pre_buffer_s      — seconds subtracted from raw onset → go back in time
    post_buffer_s     — seconds added to raw offset → safety margin at end
    on_speech_start   — callable(start_s: float)
    on_speech_end     — callable(start_s: float, end_s: float)
    on_speech_chunk   — callable(chunk: np.ndarray)  [16 ms steps during speech]
    on_first_silence  — callable(snapshot: np.ndarray)  [first silence frame]
    """

    def __init__(
        self,
        onset_threshold  : float = 0.50,
        offset_threshold : float = 0.35,
        onset_count      : int   = 2,
        offset_count     : int   = 45,
        pre_buffer_s     : float = 2.0,
        post_buffer_s    : float = 2.0,
        on_speech_start  = None,
        on_speech_end    = None,
        on_speech_chunk  = None,
        on_first_silence = None,
    ):
        self._engine = SileroEngine(threshold=onset_threshold)
        self._offset  = offset_threshold

        self._onset_count  = onset_count
        self._offset_count = offset_count
        self._pre_s        = pre_buffer_s
        self._post_s       = post_buffer_s

        self._on_speech_start  = on_speech_start
        self._on_speech_end    = on_speech_end
        self._on_speech_chunk  = on_speech_chunk
        self._on_first_silence = on_first_silence

        # Runtime state
        self._speaking         : bool  = False
        self._onset_buf        : int   = 0   # consecutive-above-threshold counter
        self._silence_count    : int   = 0   # consecutive-below-offset counter
        self._confirmed_start_s: float = 0.0 # start_s passed to callbacks (pre-buf applied)
        self._first_sil_fired  : bool  = False
        self._speech_chunks    : list  = []  # accumulated for on_first_silence snapshot
        self._leftover         : np.ndarray = np.empty(0, dtype=np.float32)

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, audio_16k: np.ndarray, current_time_s: float) -> None:
        """
        Feed a chunk of 16 kHz float32 audio.

        current_time_s is the session-elapsed time at the END of this chunk
        (i.e. the timestamp of the last sample).  VADOracle uses this to
        compute per-step absolute timestamps for speech start/end callbacks.

        Fires on_speech_start / on_speech_end / on_speech_chunk / on_first_silence
        as appropriate.  Never returns audio or utterances.
        """
        # Stitch leftover from previous call (same pattern as StreamingVAD)
        if self._leftover.size:
            audio_16k = np.concatenate([self._leftover, audio_16k])

        total_samples = len(audio_16k)
        pos = 0

        while pos + _WINDOW <= total_samples:
            window = audio_16k[pos : pos + _WINDOW]
            prob, is_speech_raw = self._engine.process_frame(window)

            # Absolute time at the END of this step.
            # current_time_s is the time of the last sample in audio_16k;
            # subtract the samples remaining after this step.
            remaining = total_samples - (pos + _STEP)
            step_end_time = current_time_s - max(0, remaining) / _SR

            step = audio_16k[pos : pos + _STEP]

            if self._speaking:
                # ── Speaking: accumulate chunk + track silence ────────────
                self._speech_chunks.append(step)

                if self._on_speech_chunk is not None:
                    try:
                        self._on_speech_chunk(step)
                    except Exception:
                        pass

                below_offset = prob < self._offset

                if below_offset:
                    self._silence_count += 1

                    # First silence frame: fire early-STT callback so the
                    # caller can start background Whisper while the full
                    # silence window elapses (same optimisation as before).
                    if self._silence_count == 1 and not self._first_sil_fired:
                        self._first_sil_fired = True
                        if self._on_first_silence is not None:
                            snapshot = (np.concatenate(self._speech_chunks)
                                        if self._speech_chunks
                                        else np.empty(0, dtype=np.float32))
                            try:
                                self._on_first_silence(snapshot)
                            except Exception:
                                pass

                    if self._silence_count >= self._offset_count:
                        # Confirmed speech end.
                        # Add post_buffer_s so the extracted window includes
                        # any trailing audio after the last word.
                        end_s = step_end_time + self._post_s
                        self._speaking = False
                        self._silence_count = 0
                        self._onset_buf     = 0
                        self._first_sil_fired = False
                        self._speech_chunks   = []
                        if self._on_speech_end is not None:
                            try:
                                self._on_speech_end(self._confirmed_start_s, end_s)
                            except Exception:
                                pass
                else:
                    # Speech resumed — reset silence counter
                    self._silence_count = 0

            else:
                # ── Not speaking: onset hysteresis ────────────────────────
                if is_speech_raw:
                    self._onset_buf += 1
                else:
                    self._onset_buf = max(0, self._onset_buf - 1)

                if self._onset_buf >= self._onset_count:
                    # Onset confirmed.  Compute the raw onset time: the step
                    # where the (onset_count)th frame fired was this step; the
                    # first speech frame was onset_count-1 steps ago.
                    raw_onset_s = step_end_time - (self._onset_count - 1) * _STEP / _SR

                    # Apply pre-buffer: go back in time by pre_buffer_s so the
                    # extracted window includes audio before VAD fired.
                    # This is the KEY fix: confirmed_start_s < raw_onset_s,
                    # so RollingAudioBuffer.read_window() retrieves audio from
                    # before the first word was detected.
                    confirmed_start_s = max(0.0, raw_onset_s - self._pre_s)
                    self._confirmed_start_s = confirmed_start_s

                    self._speaking        = True
                    self._onset_buf       = 0
                    self._silence_count   = 0
                    self._first_sil_fired = False
                    self._speech_chunks   = []

                    if self._on_speech_start is not None:
                        try:
                            self._on_speech_start(confirmed_start_s)
                        except Exception:
                            pass

            pos += _STEP

        # Keep remainder for next call (same as StreamingVAD)
        self._leftover = audio_16k[pos:].copy()

    def reset(self) -> None:
        """
        Reset LSTM hidden states and all runtime state.
        Call on voice-state transitions to prevent bleed between turns.
        """
        self._engine.reset_states()
        self._speaking         = False
        self._onset_buf        = 0
        self._silence_count    = 0
        self._first_sil_fired  = False
        self._speech_chunks    = []
        self._leftover         = np.empty(0, dtype=np.float32)
        # _confirmed_start_s intentionally preserved (used if on_speech_end
        # fires across a reset boundary, which shouldn't happen in practice)

    @property
    def is_speaking(self) -> bool:
        """True while a speech segment is being tracked (not yet ended)."""
        return self._speaking
