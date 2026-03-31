from faster_whisper import WhisperModel


class WhisperEngine:
    def __init__(self, model_name="small", device="cpu", compute_type="int8", language="en"):
        self.language = language

        # Force valid compute type
        if device == "cpu":
            compute_type = "int8"
        else:
            compute_type = "float16"

        print(f"Loading Whisper model '{model_name}' on '{device}' with compute_type='{compute_type}'")

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )

        print("Model loaded.")

    def transcribe(self, wav_path, language=None, word_timestamps=False):
        lang = language if language else self.language

        segments, info = self.model.transcribe(
            wav_path,
            language=lang,
            word_timestamps=word_timestamps
        )

        text = " ".join(
            seg.text.strip()
            for seg in segments
            if seg.text and seg.text.strip()
        )

        return text

    def set_language(self, lang_code: str):
        self.language = lang_code
        print(f"Language set to: {lang_code}")
