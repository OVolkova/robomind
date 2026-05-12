import io
from collections import deque

import soundfile as sf
import torchaudio

from robomind.llm_client import PERFORM_ACTION_TOOL, SYSTEM_PROMPT, create_llm_client
from robomind.speech_to_speech import SpeechToText, TextToSpeech


class V2Processor:
    def __init__(self, provider: str | None = None, model: str | None = None):
        print("Loading V2 models...")
        self.speech_to_text = SpeechToText()
        print("  Loaded SpeechToText (Whisper)")
        self.text_to_speech = TextToSpeech()
        print("  Loaded TextToSpeech (Kokoro)")
        self.llm = create_llm_client(provider=provider, model=model)
        print(f"  Loaded LLM client (provider=openai, model={self.llm.model})")

        self.history: deque[dict] = deque(maxlen=10)

    def process(
        self, audio_bytes: bytes, current_action: str = "balance"
    ) -> tuple[io.BytesIO, str, str | None]:
        # Speech → text
        signal, frequency = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = signal.numpy()[0]
        user_text = self.speech_to_text(waveform, frequency)
        print(f"[v2] STT: {user_text!r}")

        # Build message list: system + rolling history + new user turn
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append(
            {
                "role": "user",
                "content": f"[Current robot action: {current_action}]\n{user_text}",
            }
        )

        # LLM call
        response_text, action_key = self.llm.complete(
            messages, tools=[PERFORM_ACTION_TOOL]
        )
        print(f"[v2] LLM: text={response_text!r}, action={action_key!r}")

        # Update history with clean text (no action annotation)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text or ""})

        # Text → speech
        spoken_text = response_text.strip() or "Okay."
        out_waveform, out_freq = self.text_to_speech.generate(spoken_text)

        buf = io.BytesIO()
        sf.write(buf, out_waveform, out_freq, format="wav", subtype="PCM_16")
        buf.seek(0)

        serial_action = f"k{action_key}" if action_key else None
        return buf, response_text or "", serial_action
