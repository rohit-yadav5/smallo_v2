# classifier/constants.py

INTERRUPT_KEYWORDS = [
    "stop",
    "wait",
    "hold on",
    "pause",
    "don't continue",
    "listen",
    "bot stop",
    "one sec",
    "i want to say",
    "let me speak",
    "can i say",
    "stop talking",
    "shut up",   # optional, depends on your use case
]

# If ASR returns these → they are irrelevant
IGNORE_PATTERNS = [
    "close the door",
    "what are you doing",
    "bro",
    "hmm",
    "random",
]
