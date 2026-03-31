# listener.py
import queue
import sys
import time
import uuid
import wave
from datetime import datetime
import sounddevice as sd
import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # frame size for VAD: 10/20/30 ms allowed
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
SAMPLE_WIDTH = 2  # bytes (int16)

class AudioSegment:
    def __init__(self, filename: str, start_time: float, end_time: float):
        self.filename = filename
        self.start_time = start_time
        self.end_time = end_time

class Listener:
    """
    Listens to microphone, uses webrtcvad to detect speech.
    Produces WAV files for each voice segment.
    """

    def __init__(self, out_dir="recordings", aggressiveness=2, silence_limit=1.0, padding_ms=300):
        """
        aggressiveness: 0-3 (3 is most aggressive — fewer false positives)
        silence_limit: seconds of silence to consider segment ended
        padding_ms: keep additional ms before and after for safety
        """
        self.out_dir = out_dir
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_limit = silence_limit
        self.padding_ms = int(padding_ms)
        self.q = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Stream status:", status, file=sys.stderr)
        # indata is float32; convert to int16 little endian
        audio16 = np.frombuffer(indata, dtype=np.int16)
        self.q.put(audio16.tobytes())


    def start_stream(self):
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype="int16",
            channels=1,
            callback=self._callback
        )
        self.stream.start()
        print("Microphone stream started.\n")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Microphone stream stopped.")

    def _frame_generator(self):
        """
        Yield raw bytes frames of FRAME_DURATION_MS
        """
        while True:
            chunk = self.q.get()
            if chunk is None:
                return
            yield chunk

    def run(self, on_segment_callback):
        """
        Main loop. When a voice segment completes, writes WAV and calls on_segment_callback(path).
        """
        print("Listener running. Speak to create segments (Ctrl+C to quit).")
        self.start_stream()
        buffer = bytearray()
        voiced_frames = []
        in_speech = False
        last_voice_time = None
        segment_start_time = None

        try:
            for raw in self._frame_generator():
                # raw corresponds exactly to one frame (frame size defined in stream)
                is_speech = self.vad.is_speech(raw, SAMPLE_RATE)
                now = time.time()

                if is_speech:
                    if not in_speech:
                        # speech just started
                        in_speech = True
                        segment_start_time = now
                        # keep a little padding of previous frames if needed
                        voiced_frames = []
                    voiced_frames.append(raw)
                    last_voice_time = now
                else:
                    if in_speech:
                        # check silence duration
                        silence = now - (last_voice_time or now)
                        if silence >= self.silence_limit:
                            # segment ended
                            filename = f"{self.out_dir}/segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
                            self._save_wav(filename, voiced_frames)
                            seg = AudioSegment(filename, segment_start_time, now)
                            on_segment_callback(seg)
                            in_speech = False
                            voiced_frames = []
                            last_voice_time = None
                    else:
                        # not in speech and not starting; ignore
                        pass
        except KeyboardInterrupt:
            print("Stopping listener (KeyboardInterrupt).")
        finally:
            self.stop_stream()

    def _save_wav(self, filename, frames_bytes):
        # frames_bytes: list of raw bytes (int16)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames_bytes))
        wf.close()
        # print("Saved segment:", filename) # Optional logging to tell the name and location of the saved file (keeping it off for now)
