import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import io

from robomind.speech_to_speech import EOT_WORD, SpeechToSpeech


sts = SpeechToSpeech()


def test_speech_to_speech():
    print("Generating...")
    microphone = sr.Microphone()
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1

    for i in range(0, 3):
        with microphone as source:
            print("Talk...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
            print("... stop talking")
        signal, frequency = sf.read(io.BytesIO(audio.get_wav_data()))

        output_frequency, output_signal = sts.speech_to_speech(frequency, signal)

        memory_file = io.BytesIO()
        sf.write(memory_file, output_signal, output_frequency, format="wav")
        sound = AudioSegment.from_file(memory_file, format="wav")
        play(sound)

    print()
    for i, turn in enumerate(sts.text_context.split(EOT_WORD)):
        print(f"{('me' if i % 2 == 0 else 'model')}: {turn}")
