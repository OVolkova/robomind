import io

import soundfile as sf
import torchaudio

from robomind.robomind.speech_to_speech import SpeechToText, TextToSpeech, TextToText


class V2Processor:
    def __init__(self, provider: str | None = None, model: str | None = None):
        print("Loading V2 models...")
        self.speech_to_text = SpeechToText()
        print("  Loaded SpeechToText (Whisper)")
        self.text_to_speech = TextToSpeech()
        print("  Loaded TextToSpeech (Kokoro)")
        self.text_to_text = TextToText(provider=provider, model=model)
        print("  Loaded TextToText (LLM)")

    def process(
        self, audio_bytes: bytes, current_action: str = "balance"
    ) -> tuple[io.BytesIO, str, str | None]:
        # Speech → text
        signal, frequency = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = signal.numpy()[0]
        user_text = self.speech_to_text(waveform, frequency)
        print(f"[v2] STT: {user_text!r}")

        # LLM call (history management handled inside TextToText)
        response_text, action_key = self.text_to_text.generate(user_text, current_action)

        # Text → speech
        spoken_text = response_text.strip() or "Okay."
        out_waveform, out_freq = self.text_to_speech.generate(spoken_text)

        buf = io.BytesIO()
        sf.write(buf, out_waveform, out_freq, format="wav", subtype="PCM_16")
        buf.seek(0)

        serial_action = f"k{action_key}" if action_key else None
        return buf, response_text or "", serial_action
