import sounddevice as sd
import soundfile as sf
import os
import time

# HARD-CODED save location
RECORDINGS_DIR = "/Users/rohit/code/6hats/vad/vad_engine_silero/recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def record_audio():
    device = sd.default.device[0]
    sample_rate = int(sd.query_devices(device)["default_samplerate"])

    print("Input device:", device)
    print("Sample rate:", sample_rate)
    print("Recording... Press Ctrl+C to stop")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

    with sf.SoundFile(
        file_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as f:

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            # mono + float32
            f.write(indata[:, 0].astype("float32"))

        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            callback=callback,
        ):
            while True:
                time.sleep(0.1)


if __name__ == "__main__":
    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nRecording stopped.")