"""
stt/vad.py — Silero VAD streaming processor.

Processes a continuous stream of 16 kHz float32 audio chunks and emits
complete speech utterances when a voice segment ends.

Architecture
────────────
Browser (AudioWorklet, 4096-sample chunks at browser SR)
    ↓  WebSocket binary  [uint32 SR][float32[] samples]
_vad_loop  (resample to 16 kHz → feed here)
    ↓  complete utterance (numpy float32 @ 16 kHz)
_speech_queue  →  pipeline  →  Whisper
"""
import collections
import time
from typing import Optional

import numpy as np
import torch
from silero_vad import load_silero_vad

# Silero VAD requires exactly 512 samples per inference call at 16 kHz (32 ms).
_WINDOW   = 512
_SR       = 16000


class StreamingVAD:
    """
    Stateful streaming VAD wrapper around Silero VAD.

    Call `process(chunk_16k)` with each incoming 16 kHz chunk.
    When a complete utterance is detected, it returns a numpy array;
    otherwise it returns None.

    Parameters
    ──────────
    onset_threshold   — probability above which speech is considered started
    offset_threshold  — probability below which speech is considered to end
    silence_ms        — consecutive silence (ms) before the utterance is sealed
    min_speech_ms     — minimum speech duration to keep (ignores noise blips)
    max_speech_s      — force-flush if utterance exceeds this duration
    pre_pad_ms        — pre-speech audio to prepend (captures word beginnings)
    """

    def __init__(
        self,
        onset_threshold:  float = 0.50,
        offset_threshold: float = 0.35,
        silence_ms:       int   = 600,
        min_speech_ms:    int   = 250,
        max_speech_s:     int   = 30,
        pre_pad_ms:       int   = 200,
    ):
        self._model   = load_silero_vad()
        self._onset   = onset_threshold
        self._offset  = offset_threshold

        # Convert ms/s thresholds to VAD window counts
        self._silence_win  = max(1, int(_SR * silence_ms   / 1000 / _WINDOW))
        self._min_win      = max(1, int(_SR * min_speech_ms / 1000 / _WINDOW))
        self._max_win      =        int(_SR * max_speech_s        / _WINDOW)
        self._pre_cap      = max(1, int(_SR * pre_pad_ms   / 1000 / _WINDOW))

        # Pre-speech ring buffer (captures audio just before onset)
        self._pre:     collections.deque = collections.deque(maxlen=self._pre_cap)

        # Active speech accumulator
        self._speech:  list[np.ndarray] = []
        self._n_win:   int = 0            # windows accumulated in _speech
        self._silence: int = 0            # consecutive sub-offset windows

        self._speaking: bool = False

        # Leftover samples when a chunk doesn't divide evenly by _WINDOW
        self._leftover: np.ndarray = np.empty(0, dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, audio_16k: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed a chunk of 16 kHz float32 audio.

        Returns a complete utterance (float32 array @ 16 kHz) when speech
        ends, or None if still accumulating / waiting for speech.
        """
        # Prepend any leftover samples from the previous call
        if self._leftover.size:
            audio_16k = np.concatenate([self._leftover, audio_16k])

        result: Optional[np.ndarray] = None
        pos = 0

        while pos + _WINDOW <= len(audio_16k):
            window = audio_16k[pos : pos + _WINDOW]
            prob   = self._infer(window)

            if self._speaking:
                self._speech.append(window)
                self._n_win += 1

                if prob < self._offset:
                    self._silence += 1
                    if self._silence >= self._silence_win:
                        result = self._flush()
                else:
                    self._silence = 0   # reset silence clock on voice activity

                # Safety: force-flush on max duration
                if self._n_win >= self._max_win:
                    result = self._flush(forced=True)

            else:
                # Keep a short ring of pre-speech audio for padding
                self._pre.append(window)

                if prob >= self._onset:
                    self._speaking = True
                    self._silence  = 0

                    # Seed speech buffer with pre-speech ring (word start padding)
                    self._speech = list(self._pre) + [window]
                    self._n_win  = len(self._pre) + 1
                    self._pre.clear()

                    print(f"  [vad] ▶  speech start  prob={prob:.3f}")

            pos += _WINDOW

        # Save leftover samples for next call
        self._leftover = audio_16k[pos:].copy()
        return result

    def reset(self):
        """Reset all state — call between pipeline turns or after a long silence."""
        self._model.reset_states()
        self._pre.clear()
        self._speech  = []
        self._n_win   = 0
        self._silence = 0
        self._speaking = False
        self._leftover = np.empty(0, dtype=np.float32)

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # ── Private helpers ───────────────────────────────────────────────────

    def _infer(self, window: np.ndarray) -> float:
        """Run one Silero VAD inference on a 512-sample window."""
        with torch.no_grad():
            tensor = torch.from_numpy(window).unsqueeze(0)   # [1, 512]
            return float(self._model(tensor, _SR).detach())

    def _flush(self, forced: bool = False) -> np.ndarray:
        """Seal the current speech segment and return it (or discard if too short)."""
        audio = np.concatenate(self._speech) if self._speech else np.empty(0, dtype=np.float32)
        dur   = len(audio) / _SR

        if self._n_win >= self._min_win:
            tag = "forced" if forced else "end"
            print(f"  [vad] ■  speech {tag}  {dur:.2f}s")
        else:
            print(f"  [vad] ✗  too short ({dur:.2f}s < min) — discarding")
            audio = np.empty(0, dtype=np.float32)   # signal: discard

        self._speech   = []
        self._n_win    = 0
        self._silence  = 0
        self._speaking = False
        return audio
