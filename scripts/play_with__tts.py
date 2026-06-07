from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import io

from robomind.robomind.speech_to_speech import TextToSpeech


tts = TextToSpeech()


def test_speech_to_speech(text):
    output_signal, output_frequency = tts.generate(text)

    memory_file = io.BytesIO()
    sf.write(memory_file, output_signal, output_frequency, format="wav")
    sound = AudioSegment.from_file(memory_file, format="wav")
    play(sound)


if __name__ == "__main__":
    t = """

Example One: 

"""

    for x in [
        "DISCUSSION QUESTIONS",
    ]:
        t = t.replace(x, x.lower())
    t = t.replace("\n", " ")
    for ttt in t.split("."):
        for tttt in ttt.split("?"):
            if tttt.strip():
                test_speech_to_speech(tttt)
