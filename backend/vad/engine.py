"""
vad/engine.py — Clean Silero VAD inference engine.

Adapted from vad_2/vad_engine_silero/main.py.

Wraps silero-vad in a clean OOP class with a single process_frame() method.
Stateless per-frame inference — all state management is handled by the caller
(StreamingVAD) so this class can be unit-tested independently.
"""
import numpy as np
import torch
from silero_vad import load_silero_vad

_SR = 16_000
_WINDOW = 512   # Silero requires EXACTLY 512 samples at 16 kHz


class SileroEngine:
    """
    Thin, stateless wrapper around the Silero VAD ONNX model.

    process_frame(np.ndarray[512]) -> (probability: float, is_speech: bool)
    reset_states()                 -> resets Silero LSTM hidden states
    """

    def __init__(self, threshold: float = 0.50, sample_rate: int = _SR):
        self.threshold   = threshold
        self.sample_rate = sample_rate

        self._model = load_silero_vad()
        self._model.eval()

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[float, bool]:
        """
        Run one Silero inference step.

        frame  — exactly 512 float32 samples at 16 kHz
        returns (probability, is_speech)
        """
        tensor = self._to_tensor(frame)
        with torch.no_grad():
            prob = float(self._model(tensor, self.sample_rate).item())
        return prob, prob >= self.threshold

    def reset_states(self) -> None:
        """Reset Silero LSTM hidden states.  Call between utterances."""
        self._model.reset_states()

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.dtype.kind in "iu":
            frame = frame / float(np.iinfo(frame.dtype).max)
        return torch.from_numpy(frame)
