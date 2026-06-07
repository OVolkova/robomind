import logging
import re
import scipy
import torch
from kokoro import KPipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import io

import soundfile as sf
import torchaudio


from robomind.llm_client import PERFORM_ACTION_TOOL, SYSTEM_PROMPT, create_llm_client

logger = logging.getLogger(__name__)


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    logger.info("device is %s", device)
    return device


DEVICE = get_device()
WHISPER_FREQUENCY = 16000
KOKORO_VOICE_FREQ = 24000  # Kokoro's TTS model runs at 24kHz
ESP32_FREQUENCY = 16000  # Keep at 16kHz - ESP32 DAC may not support lower rates

_MAX_HISTORY = 10
_KEEP_RECENT = 4


def resample(waveform, target_freq: int, original_freq: int):
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
        logger.info("[TextToText] History summarised → %d messages kept", len(self.history))

    def generate(self, user_text: str, current_action: str = "balance") -> tuple[str, str | None]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append(
            {"role": "user", "content": f"[Current robot action: {current_action}]\n{user_text}"}
        )
        logger.info("[TextToText] Sending to LLM: %d messages, user_text=%r", len(messages), user_text)
        response_text, action_key = self.llm.complete(messages, tools=[PERFORM_ACTION_TOOL])
        logger.info("[TextToText] LLM: text=%r, action=%r", response_text, action_key)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text or ""})
        self._maybe_summarize()
        return response_text, action_key


class TextToSpeech:
    def __init__(self, model_freq: int = KOKORO_VOICE_FREQ):
        self.model_freq = model_freq
        self.pipeline = KPipeline(lang_code="b")  # <= make sure lang_code matches voice

    def generate(self, text):
        logger.info("Generating speech for text: %s", text)
        # 4️⃣ Generate, display, and save audio files in a loop.
        generator = self.pipeline(
            text,
            voice="af_heart",  # <= change voice here
            speed=1,
            split_pattern=r"\n+",
        )

        original_waveform = [a for _, _, a in generator][0]
        logger.info("Original waveform shape: %s, freq: %d", original_waveform.shape, self.model_freq)
        waveform = resample(
            original_waveform,
            original_freq=self.model_freq,
            target_freq=ESP32_FREQUENCY,
        )
        logger.info("Resampled waveform shape: %s, freq: %d", waveform.shape, ESP32_FREQUENCY)
        return waveform, ESP32_FREQUENCY
        # return original_waveform, self.original_freq,


class SpeechToText:
    def __init__(
        self,
        model_name="openai/whisper-tiny.en",
        model_freq: int = WHISPER_FREQUENCY,
        device: str = DEVICE,
    ):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.model_freq = model_freq

    def __call__(self, waveform, freq):
        logger.info("STT input: shape=%s freq=%d model_freq=%d", waveform.shape, freq, self.model_freq)
        waveform = resample(waveform, self.model_freq, freq)
        logger.info("STT resampled: shape=%s", waveform.shape)
        processed = self.processor(
            waveform, sampling_rate=self.model_freq, return_tensors="pt", return_attention_mask=True
        )
        input_features = processed.input_features.to(self.device)
        attention_mask = processed.attention_mask.to(self.device)

        # generate token ids
        predicted_ids = self.model.generate(input_features, attention_mask=attention_mask)

        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0]


class SpeechToSpeechActionProcessor:
    def __init__(self, provider: str | None = None, model: str | None = None):
        logger.info("Loading models...")
        self.speech_to_text = SpeechToText()
        logger.info("  Loaded SpeechToText (Whisper)")
        self.text_to_speech = TextToSpeech()
        logger.info("  Loaded TextToSpeech (Kokoro)")
        self.text_to_text = TextToText(provider=provider, model=model)
        logger.info("  Loaded TextToText (LLM)")

    def process(
        self, audio_bytes: bytes, current_action: str = "balance"
    ) -> tuple[io.BytesIO, str, str | None]:
        # Speech → text
        signal, frequency = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = signal.numpy()[0]
        user_text = self.speech_to_text(waveform, frequency)
        logger.info("STT: %r", user_text)

        # LLM call (history management handled inside TextToText)
        response_text, action_key = self.text_to_text.generate(user_text, current_action)

        # Text → speech
        spoken_text = response_text.strip()
        if spoken_text:
            out_waveform, out_freq = self.text_to_speech.generate(spoken_text)
            buf = io.BytesIO()
            sf.write(buf, out_waveform, out_freq, format="wav", subtype="PCM_16")
            buf.seek(0)
        else:
            buf = io.BytesIO()

        serial_action = f"k{action_key}" if action_key else None
        return buf, response_text or "", serial_action
