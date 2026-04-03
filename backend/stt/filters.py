"""
stt/filters.py — Post-processing filters for Whisper output.

Whisper (and distil-whisper) hallucinate on near-silence, background noise,
or music because the training data contains captions for those contexts.
Two complementary defences are applied:

  1. Exact-match blocklist — 28 known phantom phrases from the training data.
  2. Repetition detector — a phrase repeated 3+ times signals a looping
     hallucination (e.g. "you you you" or "thank you thank you thank you").

All functions are pure — no model dependency, easy to unit-test.
"""
import re

# ── Hallucination blocklist ───────────────────────────────────────────────────
# Comparisons are case-insensitive and strip surrounding punctuation/whitespace.
# Source: whisper GitHub issues + real-world observations on English-only models.
_HALLUCINATION_PHRASES: frozenset[str] = frozenset({
    # YouTube viewer-engagement tokens (common in training data)
    "thank you for watching",
    "thanks for watching",
    "thank you for watching!",
    "thanks for watching!",
    "please like and subscribe",
    "don't forget to subscribe",
    "subscribe to my channel",

    # Filler tokens on short noise bursts
    "you",
    "thank you",
    "thank you.",
    "thanks",
    "bye",
    "bye.",
    "goodbye",
    "you.",
    " you",

    # Common false positives on keyboard/click noise
    "hmm",
    "um",
    "uh",
    ".",
    "...",

    # Transcript artefacts — stored both with and without brackets
    # because _normalise() strips surrounding punctuation including [ ]
    "subtitles by",
    "subtitles by the",
    "[music]",
    "[applause]",
    "[laughter]",
    "(music)",
    "music",
    "applause",
    "laughter",
})

# Strip surrounding punctuation/whitespace before comparison so
# "You.", " you", and "YOU" all hit the same blocklist entry.
_PUNCT_STRIP = re.compile(r"^[\s.,!?;:\"'()\[\]]+|[\s.,!?;:\"'()\[\]]+$")


def _normalise(text: str) -> str:
    return _PUNCT_STRIP.sub("", text.lower()).strip()


def is_hallucination(text: str) -> bool:
    """Return True if text is a known hallucination or a repetition loop.

    Also returns True for empty/whitespace-only strings so callers can use
    this as a combined empty-check + hallucination gate.
    """
    if not text or not text.strip():
        return True

    norm = _normalise(text)

    if norm in _HALLUCINATION_PHRASES:
        return True

    return _is_repetition_loop(norm)


def _is_repetition_loop(text: str) -> bool:
    """Detect looping hallucinations: a chunk of 1–6 words repeated 3+ times.

    Examples caught:
        "you you you you"                            (chunk=1, reps=4)
        "thank you thank you thank you"              (chunk=2, reps=3)
        "the music plays the music plays the music plays"  (chunk=3, reps=3)

    Allows at most 1 trailing leftover word to handle minor tail fragments.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return False   # need at least 3 repetitions of a 1-word chunk

    for chunk_size in range(1, 7):
        if len(tokens) < chunk_size * 3:
            continue   # not enough tokens for 3 full repetitions
        chunk = tokens[:chunk_size]
        reps  = len(tokens) // chunk_size
        rem   = len(tokens) % chunk_size
        if rem > 1:
            continue   # too many leftover words — not a clean repetition
        if all(tokens[i * chunk_size:(i + 1) * chunk_size] == chunk
               for i in range(reps)):
            return True

    return False
