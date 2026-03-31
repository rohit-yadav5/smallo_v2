import sounddevice as sd
import soundfile as sf
import os
import time
import torch

# ---- config ----
RECORDINGS_DIR = "/Users/rohit/code/6hats/vad/vad_engine_silero/recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

PAUSED = True   # start paused until speech is detected

# debounce (in seconds)
SPEECH_ON_DELAY = 0.3
SPEECH_OFF_DELAY = 0.5

_last_speech_time = None
_last_silence_time = None

def record_audio():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )

    device = sd.default.device[0]
    sample_rate = int(sd.query_devices(device)["default_samplerate"])

    print("Input device:", device)
    print("Sample rate:", sample_rate)
    print("Recording... Press Ctrl+C to stop")

    ts = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RECORDINGS_DIR, f"recording_{ts}.wav")

    with sf.SoundFile(
        file_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="PCM_16"
    ) as f:

        def callback(indata, frames, time_info, status):
            global PAUSED, _last_speech_time, _last_silence_time

            if status:
                print(status)

            audio = indata[:, 0].astype("float32")

            # Silero model outputs speech probability directly
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            prob = model(audio_tensor).item()

            now = time.time()

            if prob > 0.5:
                _last_speech_time = now
                if PAUSED and (_last_silence_time is None or now - _last_silence_time > SPEECH_ON_DELAY):
                    PAUSED = False
                    print("[AUTO RESUME]")
            else:
                _last_silence_time = now
                if not PAUSED and (_last_speech_time is None or now - _last_speech_time > SPEECH_OFF_DELAY):
                    PAUSED = True
                    print("[AUTO PAUSE]")

            if not PAUSED:
                f.write(audio)

        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            callback=callback
        ):
            while True:
                time.sleep(0.1)


if __name__ == "__main__":
    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nRecording stopped.")