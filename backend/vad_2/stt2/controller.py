# controller.py
import os
from pathlib import Path
from listener import Listener, AudioSegment
from whisper_engine import WhisperEngine
from datetime import datetime

RECS_DIR = "recordings"

class STTController:
    def __init__(self, model_name="small", device="cpu"):
        Path(RECS_DIR).mkdir(parents=True, exist_ok=True)
        self.listener = Listener(out_dir=RECS_DIR, aggressiveness=2, silence_limit=1.0, padding_ms=300)
        self.engine = WhisperEngine(model_name=model_name, device=device)
        print("STTController initialized.")

    def _save_transcript(self, text):
        import json
        from datetime import datetime

        json_path = "/Users/rohit/code/6hats/vad_old_3/stt2" #this keeps changing

        # Determine incremental ID
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        if len(data) > 0 and "id" in data[-1]:
            next_id = data[-1]["id"] + 1
        else:
            next_id = 1

        formatted_id = f"{next_id:05d}"

        entry = {
            "id": next_id,
            "id_str": formatted_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": text
        }

        data.append(entry)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def on_segment(self, segment: AudioSegment):
        try:
            text = self.engine.transcribe(segment.filename)
            if not text.strip():
                print("No speech detected.\n")
            else:
                print(f"Transcript: {text}\n")
            self._save_transcript(text)
            # delete recording after use
            try:
                Path(segment.filename).unlink()
            except Exception as del_err:
                print(f"Could not delete file {segment.filename}: {del_err}")
        except Exception as e:
            print("Error transcribing:", e) # Optional error logging (keeping it for now)

    def run(self):
        print("Listening...")
        self.listener.run(self.on_segment)
