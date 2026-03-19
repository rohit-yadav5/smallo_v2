import sounddevice as sd
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
import collections

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2)  # 0–3 (higher = more aggressive)

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

def int16_pcm(audio):
    return (audio * 32768).astype(np.int16)

print("Listening... Speak now")

frames = []
ring_buffer = collections.deque(maxlen=20)
triggered = False

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=FRAME_SIZE
) as stream:

    while True:
        audio, _ = stream.read(FRAME_SIZE)
        pcm = int16_pcm(audio.flatten())

        is_speech = vad.is_speech(pcm.tobytes(), SAMPLE_RATE)

        if not triggered:
            ring_buffer.append((pcm, is_speech))
            num_voiced = sum(1 for _, s in ring_buffer if s)

            if num_voiced > 0.7 * ring_buffer.maxlen:
                triggered = True
                frames.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
        else:
            frames.append(pcm)
            ring_buffer.append((pcm, is_speech))
            num_unvoiced = sum(1 for _, s in ring_buffer if not s)

            if num_unvoiced > 0.8 * ring_buffer.maxlen:
                break

print("Processing...")

audio_data = np.concatenate(frames).astype(np.float32) / 32768.0

segments, _ = model.transcribe(audio_data, language="en")
text = " ".join(seg.text for seg in segments)

print("You said:", text)
