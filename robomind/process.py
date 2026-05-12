import io
import random

import torchaudio
import soundfile as sf

from robomind.speech_to_speech import SpeechToSpeech

sts = SpeechToSpeech()

RANDOM_REQUEST_TEXTS = [
    "What is volcano?",
    "Why rainbows have 7 colors?",
    "What is evolution?",
    "What is the meaning of life?",
    "What is the universe?",
    "What is the theory of relativity?",
]


def _return_wav_file(output_frequency, output_signal):
    """Return audio as WAV format (PCM) in a BytesIO stream."""
    memory_file = io.BytesIO()
    sf.write(
        memory_file, output_signal, output_frequency, format="wav", subtype="PCM_16"
    )
    memory_file.seek(0)  # Reset pointer to the beginning of the BytesIO object
    return memory_file


def process(audio_data):
    """Process the audio stream WAV and return the processed WAV as a stream."""
    # Detect format from data
    signal, frequency = torchaudio.load(io.BytesIO(audio_data))
    signal = signal.numpy()[0]

    output_frequency, output_signal = sts.speech_to_speech(frequency, signal)

    return _return_wav_file(output_frequency, output_signal)


# def random_answer():
#     request_text = random.choice(RANDOM_REQUEST_TEXTS)

#     output_frequency, output_signal = sts.answer_to_text_with_speech(request_text)

#     return _return_wav_file(output_frequency, output_signal)


def random_answer():
    output_signal, output_frequency = sts.text_to_speech.generate("Hi")

    return _return_wav_file(output_frequency, output_signal)
