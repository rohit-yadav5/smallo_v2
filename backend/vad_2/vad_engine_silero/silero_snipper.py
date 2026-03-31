import torch
import soundfile as sf
import numpy as np
import os

# ---------------- CONFIG ----------------
INPUT_WAV = "/Users/rohit/code/6hats/vad/vad_engine_silero/recordings/example.wav"
OUTPUT_WAV = "/Users/rohit/code/6hats/vad/vad_engine_silero/recordings/example_speech_only.wav"
TARGET_SR = 16000
# ----------------------------------------


def main():
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )

    get_speech_timestamps = utils[0]

    # Load audio with soundfile (no torchcodec dependency)
    audio, sr = sf.read(INPUT_WAV)
    print("Original sample rate:", sr)

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)

    # Resample to 16kHz (Silero requirement)
    if sr != TARGET_SR:
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            scale_factor=TARGET_SR / sr,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        sr = TARGET_SR

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform.squeeze(0),
        model,
        sampling_rate=TARGET_SR
    )

    if not speech_timestamps:
        print("No speech detected.")
        return

    print(f"Detected {len(speech_timestamps)} speech segments")

    # Cut and collect speech segments
    speech_chunks = []
    for seg in speech_timestamps:
        start = seg["start"]
        end = seg["end"]
        speech_chunks.append(waveform[:, start:end])

    # Concatenate all speech
    speech_audio = torch.cat(speech_chunks, dim=1)

    # Save output
    sf.write(OUTPUT_WAV, speech_audio.squeeze(0).numpy(), TARGET_SR)
    print("Saved speech-only file to:")
    print(OUTPUT_WAV)


if __name__ == "__main__":
    main()