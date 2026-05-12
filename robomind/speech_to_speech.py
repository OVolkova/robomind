import scipy
import tiktoken
import torch
from collections import deque
from kokoro import KPipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from robomind.model import GPT
# from robomind.stm import ShortTermMemory


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
EOT_WORD = "<|endoftext|>"


def resample(waveform, target_freq, original_freq: int = FREQUENCY):
    if target_freq == original_freq:
        return waveform
    return scipy.signal.resample(
        waveform, int(waveform.shape[0] * target_freq / original_freq)
    )


class TextToText:
    def __init__(
        self,
        checkpoint_path="/Users/olly/Documents/projects/llms/gpt2/model/model_wow.pt",
        device=DEVICE,
    ):
        self.device = device
        self.model = self.load_model(checkpoint_path)

        self.encoder = tiktoken.get_encoding("gpt2")
        self.EOT = self.encoder._special_tokens["<|endoftext|>"]  # end of text token

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("mps"), weights_only=False
        )
        print(
            f" loaded model from step {checkpoint['step']} with validation loss {checkpoint['val_loss']}"
        )

        model = GPT(config=checkpoint["config"])
        # Remove "_orig_mod." prefix from keys
        new_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
        }

        # Load modified state dict into model
        model.load_state_dict(new_state_dict, strict=False)

        model.to(self.device)
        model.eval()
        return model

    def generate(self, text):
        # convert text to list of tokens
        tokens = self.encoder.encode_ordinary(text)
        # convert tokens to tensor
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        # generate new tokens
        x = self.model.generate_till_eot(x, eot_token=self.EOT).detach().tolist()[0]
        # decode new tokens starting from position len(tokens) to text
        decoded = self.encoder.decode(x[len(tokens) :])
        return decoded


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


class SpeechToSpeech:
    def __init__(self):
        print("Loading models...")
        self.text_to_text = TextToText()
        print("Loaded text_to_text")
        self.speech_to_text = SpeechToText()
        print("Loaded speech_to_text")
        self.text_to_speech = TextToSpeech()
        print("Loaded text_to_speech")

        self.text_context = deque([])
        self.max_context_length = 5

    def reset_context(self):
        self.text_context = deque([])

    def speech_to_speech(self, input_frequency, input_signal):
        print("transcribing audio...")
        request_text = self.speech_to_text(input_signal, input_frequency)

        output_frequency, output_signal = self.answer_to_text_with_speech(request_text)
        return output_frequency, output_signal

    def answer_to_text_with_speech(self, request_text):
        self.text_context.append(request_text)
        print(f" -- request_text - {request_text}")

        print("generating text...")
        response_text = ""
        while len(response_text.strip()) == 0:
            response_text = self.text_to_text.generate(EOT_WORD.join(self.text_context))
        response_text = response_text.replace("\n", " ")
        response_text = response_text.replace(EOT_WORD, " ")
        self.text_context.append(response_text)
        if len(self.text_context) > self.max_context_length:
            self.text_context.popleft()
        print(f" -- response_text - {response_text}")
        print(f" -- full context - {self.text_context}")

        print("generating audio...")
        output_signal, output_frequency = self.text_to_speech.generate(response_text)
        print("done")
        return output_frequency, output_signal
