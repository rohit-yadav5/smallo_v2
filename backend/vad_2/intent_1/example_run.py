from hybrid_classifier import HybridIntentClassifier

clf = HybridIntentClassifier()

import json
import os

STT_PATH = "/Users/rohit/code/6hats/vad/intent_1/stt.json"

def load_stt_entries():
    if os.path.exists(STT_PATH):
        with open(STT_PATH, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_remaining_entries(entries):
    with open(STT_PATH, "w") as f:
        json.dump(entries, f, indent=4)

entries = load_stt_entries()

total_latency = 0
count = 0

for entry in entries:
    text = entry.get("text", "").strip()
    if not text:
        continue

    out = clf.predict(text)
    intent = out.get("result", out.get("label", "UNKNOWN"))
    print(f"[{intent}]: {text}")
    count += 1

# After processing all entries, clear the file
save_remaining_entries([])

print("\nProcessed all STT entries and cleared stt.json.")


    #     # Only print ML-based decisions
    # if out.get("source") == "ml":
    #     print(f"{t} -> {out.get('result', out.get('label', 'UNKNOWN'))}")
    #     continue


#        # for ml base results
# for t in tests:
#     result = clf.predict(t)
#     if result.get("source") == "ml":
#         print("\nText:", t)
#         print(result)

#        # for rule base results
# for t in tests:
#     result = clf.predict(t)
#     print("\nText:", t)
#     print(result)


#         # for both rule and ml results
# for t in tests:
#     result = clf.predict(t)
#     print("\nText:", t)
#     print(result)