import io

import requests

import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play


if __name__ == "__main__":
    print("Generating...")
    microphone = sr.Microphone()
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1

    for i in range(0, 10):
        # Simplified: WAV -> Server -> WAV (no MP3!)
        with microphone as source:
            print("Talk...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
            print("... stop talking")

        # Get WAV data from recognizer
        wav_data = audio.get_wav_data()

        # Send POST request with raw WAV data
        response = requests.post(
            "http://127.0.0.1:7777/process",
            data=wav_data,
            headers={"Content-Type": "audio/wav"},
        )

        # Play WAV response
        memory_file = io.BytesIO(response.content)
        sound = AudioSegment.from_file(memory_file, format="wav")
        play(sound)
