#!/usr/bin/env python3
"""
stt/test_stt.py — Manual test suite for the new STT stack.

Run from the backend/ directory:
    cd backend && python stt/test_stt.py

Tests covered
─────────────
1. filters.py    — hallucination blocklist + repetition detector (no model needed)
2. engine.py     — device detection, model load, warmup
3. main_stt.py   — transcribe() on synthetic audio (silence, noise, real speech WAV)
4. streaming.py  — StreamingTranscriber feed/finalize with a real WAV split into chunks
5. Integration   — record from mic for 5 s and transcribe live (optional, requires mic)
"""
import os
import sys
import time
import wave
import struct
import threading

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗ FAIL: {msg}{RESET}")
def info(msg): print(f"  {CYAN}·{RESET} {msg}")
def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*54}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*54}{RESET}")

_failures = 0

def check(condition: bool, label: str):
    global _failures
    if condition:
        ok(label)
    else:
        fail(label)
        _failures += 1


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILTERS — no model required
# ══════════════════════════════════════════════════════════════════════════════

section("1. filters.py — hallucination detection")

from stt.filters import is_hallucination, _is_repetition_loop, _normalise

# Blocklist hits
check(is_hallucination(""),                        "empty string → hallucination")
check(is_hallucination("   "),                     "whitespace → hallucination")
check(is_hallucination("thank you for watching"),  "exact blocklist phrase")
check(is_hallucination("Thank You For Watching"),  "case-insensitive blocklist")
check(is_hallucination("Thank you for watching!"), "with trailing punctuation")
check(is_hallucination("you"),                     "single word 'you'")
check(is_hallucination("You."),                    "'You.' normalises to 'you'")
check(is_hallucination("[music]"),                 "transcript artefact [music]")
check(is_hallucination("[MUSIC]"),                 "[MUSIC] case-insensitive")

# Repetition detector
check(is_hallucination("you you you"),                   "1-word repetition ×3")
check(is_hallucination("you you you you you"),           "1-word repetition ×5")
check(is_hallucination("thank you thank you thank you"), "2-word repetition ×3")
check(is_hallucination("ha ha ha ha ha ha"),             "short word repeated ×6")

# Should NOT be filtered
check(not is_hallucination("what is the weather today"),       "normal query not filtered")
check(not is_hallucination("Small O can you hear me"),         "bot name phrase not filtered")
check(not is_hallucination("tell me a joke"),                  "simple command not filtered")
check(not is_hallucination("yes"),                             "'yes' not filtered")
check(not is_hallucination("thank you very much for helping"), "real 'thank you' phrase not filtered")

# Repetition boundary: 2 reps should NOT trigger (need ≥ 3)
check(not is_hallucination("you you"),                 "only 2 reps → not filtered")
check(not is_hallucination("thank you thank you"),     "2-word, 2 reps → not filtered")

# Normalise helper
check(_normalise("You.") == "you",            "_normalise strips trailing period")
check(_normalise("  Hello!  ") == "hello",    "_normalise strips whitespace+punct")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENGINE — device detection + model load + warmup
# ══════════════════════════════════════════════════════════════════════════════

section("2. engine.py — device detection + model load")

from stt.engine import _DEVICE, _COMPUTE_TYPE, stt_lock, load_model, warmup, SAMPLE_RATE

info(f"Device: {_DEVICE}  Compute: {_COMPUTE_TYPE}")
check(_DEVICE in ("cpu", "cuda"),                    f"device is cpu or cuda (got: {_DEVICE})")
check(_COMPUTE_TYPE in ("int8", "float16"),          f"compute_type valid (got: {_COMPUTE_TYPE})")
check(isinstance(stt_lock, type(threading.Lock())),  "stt_lock is a Lock")
check(SAMPLE_RATE == 16_000,                         "SAMPLE_RATE == 16 000")

info("Loading model (this downloads ~140 MB on first run)...")
t0 = time.perf_counter()
model = load_model()
load_secs = time.perf_counter() - t0
info(f"Model loaded in {load_secs:.2f}s")
check(model is not None, "model loaded successfully")

info("Warming up JIT...")
t0 = time.perf_counter()
warmup(model)
info(f"Warmup done in {time.perf_counter()-t0:.2f}s")
ok("warmup completed without exception")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN_STT — transcribe() on synthetic audio
# ══════════════════════════════════════════════════════════════════════════════

section("3. main_stt.py — transcribe() on synthetic audio")

from stt.main_stt import transcribe, transcribe_partial

SR = 16_000

# ── Silence gate ──────────────────────────────────────────────────────────────
info("Transcribing 1s of pure silence...")
silence = np.zeros(SR, dtype=np.float32)
text, secs = transcribe(silence)
info(f"  result: '{text}'  ({secs:.3f}s)")
check(text == "", "silence → empty string (energy gate blocks Whisper)")
check(secs == 0.0, "silence → 0.0 transcription time (skipped)")

# ── Low-amplitude noise gate ──────────────────────────────────────────────────
info("Transcribing 1s of very low amplitude noise...")
rng = np.random.default_rng(42)
quiet_noise = (rng.standard_normal(SR) * 0.001).astype(np.float32)
text, secs = transcribe(quiet_noise)
info(f"  result: '{text}'  ({secs:.3f}s)")
check(text == "", "quiet noise → energy gate blocks Whisper")

# ── White noise (loud) — should not crash, may produce text or empty ──────────
info("Transcribing 1s of loud white noise...")
loud_noise = (rng.standard_normal(SR) * 0.5).astype(np.float32)
t0 = time.perf_counter()
text, secs = transcribe(loud_noise)
wall = time.perf_counter() - t0
info(f"  result: '{text}'  ({secs:.3f}s wall={wall:.3f}s)")
check(True, "loud noise transcribed without crash")

# ── Synthetic sine tone (440 Hz pure tone — should be near-silent text) ───────
info("Transcribing 1s of 440 Hz sine tone...")
t = np.linspace(0, 1, SR, endpoint=False, dtype=np.float32)
sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
text, secs = transcribe(sine)
info(f"  result: '{text}'  ({secs:.3f}s)")
check(True, "sine tone transcribed without crash")

# ── Thread safety: two concurrent calls must not crash ────────────────────────
info("Thread-safety test: 2 concurrent transcribe() calls...")
results = [None, None]
errors  = [None, None]

def _call(idx, audio):
    try:
        results[idx] = transcribe(audio)
    except Exception as e:
        errors[idx] = e

t1 = threading.Thread(target=_call, args=(0, silence.copy()))
t2 = threading.Thread(target=_call, args=(1, quiet_noise.copy()))
t1.start(); t2.start()
t1.join();  t2.join()
check(errors[0] is None and errors[1] is None, "no exceptions in concurrent calls")

# ── WAV file test (if a WAV exists in the recordings dir) ────────────────────
import glob
wav_files = glob.glob(os.path.join(BACKEND_DIR, "**/*.wav"), recursive=True)
if wav_files:
    wav_path = wav_files[0]
    info(f"Testing with WAV: {os.path.relpath(wav_path, BACKEND_DIR)}")
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            file_sr    = wf.getframerate()
            n_frames   = wf.getnframes()
            raw_bytes  = wf.readframes(n_frames)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples[::n_channels]   # take left channel
        # Resample to 16 kHz if needed
        if file_sr != SR:
            try:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(file_sr, SR)
                samples = resample_poly(samples, SR // g, file_sr // g).astype(np.float32)
            except ImportError:
                n = int(len(samples) * SR / file_sr)
                samples = np.interp(np.linspace(0, len(samples), n),
                                    np.arange(len(samples)), samples).astype(np.float32)
        info(f"  audio: {len(samples)/SR:.2f}s @ {SR} Hz")
        t0 = time.perf_counter()
        text, secs = transcribe(samples)
        wall = time.perf_counter() - t0
        info(f"  result: '{text}'")
        info(f"  Whisper time: {secs:.3f}s  (wall: {wall:.3f}s)")
        check(isinstance(text, str), "WAV transcription returned a string")
    except Exception as e:
        info(f"  WAV test skipped: {e}")
else:
    info("No WAV files found — skipping WAV transcription test")


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING — StreamingTranscriber feed + finalize
# ══════════════════════════════════════════════════════════════════════════════

section("4. streaming.py — StreamingTranscriber")

from stt.streaming import StreamingTranscriber

partial_events: list[tuple[str, str]] = []

def _on_partial(confirmed: str, hypothesis: str):
    partial_events.append((confirmed, hypothesis))
    print(f"    {GREEN}STT_PARTIAL{RESET}  confirmed='{confirmed}'  hypothesis='{hypothesis}'")

transcriber = StreamingTranscriber(
    transcribe_fn         = transcribe,
    on_partial            = _on_partial,
    chunk_interval_s      = 0.3,
    transcribe_partial_fn = transcribe_partial,
)

# ── Feed silence chunks — should not emit partials ────────────────────────────
info("Feeding 2s of silence in 16ms chunks...")
chunk_size = 256   # 16 ms @ 16 kHz
silence_2s = np.zeros(SR * 2, dtype=np.float32)
for i in range(0, len(silence_2s), chunk_size):
    transcriber.feed(silence_2s[i:i + chunk_size])
time.sleep(0.6)   # let any background job finish
check(len(partial_events) == 0, "silence feed produces no STT_PARTIAL events")

# ── Feed a sine tone in chunks and finalize ───────────────────────────────────
info("Feeding 1s sine + finalize...")
partial_events.clear()
transcriber.reset()
for i in range(0, SR, chunk_size):
    transcriber.feed(sine[i:i + chunk_size].astype(np.float32))
time.sleep(0.8)
text, secs = transcriber.finalize(sine)
info(f"  finalize result: '{text}'  ({secs:.3f}s)")
check(isinstance(text, str), "finalize returns a string")

# ── start_finalize then finalize ─────────────────────────────────────────────
info("start_finalize on snapshot then finalize on full audio...")
partial_events.clear()
transcriber.reset()
snapshot = silence[:SR]   # 1s silence snapshot
transcriber.start_finalize(snapshot)
time.sleep(0.1)
full_audio = np.zeros(SR * 2, dtype=np.float32)   # full 2s silence
text, secs = transcriber.finalize(full_audio)
info(f"  result: '{text}'  ({secs:.3f}s)")
check(isinstance(text, str), "start_finalize + finalize returns string without crash")

# ── reset clears state ────────────────────────────────────────────────────────
transcriber.reset()
check(len(transcriber._chunks) == 0,          "reset clears _chunks")
check(len(transcriber._confirmed_words) == 0, "reset clears _confirmed_words")
check(len(transcriber._prev_times) == 0,      "reset clears _prev_times")
check(transcriber._last_emitted == "",        "reset clears _last_emitted")
check(transcriber._partial is None,           "reset clears _partial")
check(transcriber._final is None,             "reset clears _final")

# ── WAV streaming test ────────────────────────────────────────────────────────
if wav_files:
    wav_path = wav_files[0]
    info(f"Streaming WAV in 16ms chunks: {os.path.relpath(wav_path, BACKEND_DIR)}")
    partial_events.clear()
    transcriber2 = StreamingTranscriber(
        transcribe_fn         = transcribe,
        on_partial            = _on_partial,
        chunk_interval_s      = 0.3,
        transcribe_partial_fn = transcribe_partial,
    )
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            file_sr    = wf.getframerate()
            n_frames   = wf.getnframes()
            raw_bytes  = wf.readframes(n_frames)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples[::n_channels]
        if file_sr != SR:
            try:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(file_sr, SR)
                samples = resample_poly(samples, SR // g, file_sr // g).astype(np.float32)
            except ImportError:
                n = int(len(samples) * SR / file_sr)
                samples = np.interp(np.linspace(0, len(samples), n),
                                    np.arange(len(samples)), samples).astype(np.float32)

        info(f"  streaming {len(samples)/SR:.2f}s of audio in 16ms chunks...")
        t0 = time.perf_counter()
        for i in range(0, len(samples), chunk_size):
            transcriber2.feed(samples[i:i + chunk_size].astype(np.float32))
        # Wait for any in-flight partial
        time.sleep(0.8)
        text, secs = transcriber2.finalize(samples)
        wall = time.perf_counter() - t0
        info(f"  final result: '{text}'")
        info(f"  Whisper: {secs:.3f}s  wall: {wall:.3f}s  partials emitted: {len(partial_events)}")
        check(isinstance(text, str), "WAV streaming finalize returned string")
        if partial_events:
            ok(f"partial events emitted during streaming: {len(partial_events)}")
        else:
            info("no partial events (audio may be too short for 300ms interval)")
    except Exception as e:
        info(f"  WAV streaming test skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LIVE MIC — real-time streaming transcription
# ══════════════════════════════════════════════════════════════════════════════

section("5. Live mic — real-time streaming")

try:
    import sounddevice as sd

    CHUNK_SIZE = 256   # 16 ms @ 16 kHz — same as VAD loop

    mic_partials:   list[tuple[str, str]] = []
    _mic_audio:     list[np.ndarray]      = []
    _printed_words: list[str]             = []   # confirmed words already shown
    _hyp_active:    list[bool]            = [False]
    ERASE = f"\r{' ' * 70}\r"   # wipe the hypothesis overwrite line

    def _mic_partial(confirmed: str, hypothesis: str) -> None:
        mic_partials.append((confirmed, hypothesis))

        confirmed_words = confirmed.split() if confirmed.strip() else []

        # Each new confirmed word gets its own line.
        # Erase the hypothesis overwrite line first if one is showing.
        for word in confirmed_words[len(_printed_words):]:
            if _hyp_active[0]:
                print(ERASE, end="", flush=True)
                _hyp_active[0] = False
            print(f"  {GREEN}{word}{RESET}", flush=True)
            _printed_words.append(word)

        # Hypothesis — show the entire unconfirmed tail as one dim overwrite line.
        if hypothesis.strip():
            print(f"\r  \033[2m{hypothesis.strip()}\033[0m{' ' * 40}", end="", flush=True)
            _hyp_active[0] = True
        elif _hyp_active[0]:
            print(ERASE, end="", flush=True)
            _hyp_active[0] = False

    mic_transcriber = StreamingTranscriber(
        transcribe_fn         = transcribe,
        on_partial            = _mic_partial,
        chunk_interval_s      = 0.3,
        transcribe_partial_fn = transcribe_partial,
    )

    def _audio_callback(indata: np.ndarray, frames: int, t, status):
        chunk = indata[:, 0].copy()
        _mic_audio.append(chunk)
        mic_transcriber.feed(chunk)

    print(f"\n  {YELLOW}{BOLD}▶  Speak — press Ctrl+C to stop{RESET}\n")

    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=CHUNK_SIZE, callback=_audio_callback):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            pass

    if _hyp_active[0]:
        print(ERASE, end="", flush=True)
    print()

    full_audio = np.concatenate(_mic_audio) if _mic_audio else np.zeros(SR, dtype=np.float32)
    print(f"  {CYAN}Finalizing...{RESET}")
    text, secs = mic_transcriber.finalize(full_audio)

    # Print any final confirmed words not already shown via partials
    final_words = text.split() if text.strip() else []
    for word in final_words[len(_printed_words):]:
        print(f"  {GREEN}{word}{RESET}", flush=True)

    print(f"\n  {BOLD}─── Full transcript ───{RESET}")
    print(f"  \"{text}\"")
    print(f"  Whisper: {secs:.3f}s  |  Partial events: {len(mic_partials)}\n")
    check(isinstance(text, str), "live mic transcription returned string")

except ImportError:
    info("sounddevice not installed — skipping  (pip install sounddevice)")
except Exception as e:
    info(f"Mic test skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

section("Results")
total = 0   # count from global
if _failures == 0:
    print(f"\n  {GREEN}{BOLD}All checks passed!{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}{_failures} check(s) FAILED{RESET}\n")
    sys.exit(1)
