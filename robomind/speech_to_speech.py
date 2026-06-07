import scipy
import torch
from kokoro import KPipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from robomind.llm_client import PERFORM_ACTION_TOOL, SYSTEM_PROMPT, create_llm_client


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"device is {device}")
    return device


DEVICE = get_device()
FREQUENCY = 16000
ESP32_FREQUENCY = 16000  # Keep at 16kHz - ESP32 DAC may not support lower rates

_MAX_HISTORY = 10
_KEEP_RECENT = 4


def resample(waveform, target_freq, original_freq: int = FREQUENCY):
    if target_freq == original_freq:
        return waveform
    return scipy.signal.resample(
        waveform, int(waveform.shape[0] * target_freq / original_freq)
    )


class TextToText:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.llm = create_llm_client(provider=provider, model=model)
        self.history: list[dict] = []

    def _maybe_summarize(self) -> None:
        if len(self.history) < _MAX_HISTORY:
            return
        to_summarize = self.history[:-_KEEP_RECENT]
        recent = self.history[-_KEEP_RECENT:]
        summary_text, _ = self.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Summarise the following robot-dog conversation in 2-3 sentences. "
                        "Note topics discussed and any physical actions the robot performed."
                    ),
                },
                *to_summarize,
                {"role": "user", "content": "Summarise the conversation above briefly."},
            ]
        )
        self.history = [
            {
                "role": "system",
                "content": f"Summary of earlier conversation: {summary_text or 'Previous exchanges occurred.'}",
            },
            *recent,
        ]
        print(f"[TextToText] History summarised → {len(self.history)} messages kept")

    def generate(self, user_text: str, current_action: str = "balance") -> tuple[str, str | None]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append(
            {"role": "user", "content": f"[Current robot action: {current_action}]\n{user_text}"}
        )
        response_text, action_key = self.llm.complete(messages, tools=[PERFORM_ACTION_TOOL])
        print(f"[TextToText] LLM: text={response_text!r}, action={action_key!r}")
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text or ""})
        self._maybe_summarize()
        return response_text, action_key


class TextToSpeech:
    def __init__(self, original_freq: int = 24000):
        self.original_freq = original_freq
        self.pipeline = KPipeline(lang_code="b")  # <= make sure lang_code matches voice

    def generate(self, text):
        print(f"Generating speech for text -  {text}")
        # 4️⃣ Generate, display, and save audio files in a loop.
        generator = self.pipeline(
            text,
            voice="af_heart",  # <= change voice here
            speed=1,
            split_pattern=r"\n+",
        )

        original_waveform = [a for _, _, a in generator][0]
        print(
            f"Original waveform shape: {original_waveform.shape}, freq: {self.original_freq}"
        )
        waveform = resample(
            original_waveform,
            original_freq=self.original_freq,
            target_freq=ESP32_FREQUENCY,
        )
        print(f"Resampled waveform shape: {waveform.shape}, freq: {ESP32_FREQUENCY}")
        return waveform, ESP32_FREQUENCY
        # return original_waveform, self.original_freq,


class SpeechToText:
    def __init__(
        self,
        model_name="openai/whisper-tiny.en",
        original_freq: int = FREQUENCY,
        device: str = DEVICE,
    ):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.config.forced_decoder_ids = None
        self.original_freq = original_freq

    def __call__(self, waveform, freq):
        waveform = resample(waveform, freq, self.original_freq)
        input_features = self.processor(
            waveform, sampling_rate=self.original_freq, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device)

        # generate token ids
        predicted_ids = self.model.generate(input_features)

        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )
        return transcription[0]


