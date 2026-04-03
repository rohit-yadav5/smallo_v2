"""
vad/ — Voice Activity Detection module.

Architecture
────────────
SileroEngine   — stateless per-frame Silero VAD inference
RingBuffer     — thread-safe circular audio ring buffer
StreamingVAD   — legacy gating VAD (kept for reference; no longer in main path)
VADOracle      — NEW: timestamp-only VAD; fires start/end callbacks, never
                 accumulates audio.  Used with RollingAudioBuffer in the
                 continuous-recording pipeline.

New pipeline usage
──────────────────
from vad import VADOracle

oracle = VADOracle(
    pre_buffer_s=2.0, post_buffer_s=2.0,
    on_speech_start=..., on_speech_end=...,
)
oracle.process(chunk_16k, current_time_s)   # fires callbacks
oracle.reset()                              # on state transitions
"""
from vad.streaming   import StreamingVAD
from vad.engine      import SileroEngine
from vad.ring_buffer import RingBuffer
from vad.oracle      import VADOracle

__all__ = ["StreamingVAD", "SileroEngine", "RingBuffer", "VADOracle"]
