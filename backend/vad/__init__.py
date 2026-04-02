"""
vad/ — Voice Activity Detection module.

Architecture
────────────
SileroEngine   — stateless per-frame Silero VAD inference (vad_2 inspired)
RingBuffer     — thread-safe circular audio ring buffer   (vad_2 inspired)
StreamingVAD   — stateful streaming wrapper; public interface for the pipeline

Usage
─────
from vad import StreamingVAD

vad = StreamingVAD()
utterance = vad.process(chunk_16k)   # returns np.ndarray or None
vad.reset()                          # between sessions / after barge-in
vad.is_speaking                      # True while accumulating speech
"""
from vad.streaming   import StreamingVAD
from vad.engine      import SileroEngine
from vad.ring_buffer import RingBuffer

__all__ = ["StreamingVAD", "SileroEngine", "RingBuffer"]
