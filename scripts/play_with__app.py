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
        # TEST THAT WAV -> MP3 -> WAV -> MP3 WORKS
        with microphone as source:
            print("Talk...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
            print("... stop talking")

        # Get WAV data from recognizer
        wav_data = audio.get_wav_data()
        wav_io = io.BytesIO(wav_data)

        # Convert WAV to MP3 using pydub
        audio_segment = AudioSegment.from_file(wav_io, format="wav")
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        # Send POST request with raw MP3 data
        response = requests.post("http://127.0.0.1:7777/process",
                                 data=mp3_io.read(),
                                 headers={"Content-Type": "audio/mpeg"}
                                 )
        memory_file = io.BytesIO(response.content)
        sound = AudioSegment.from_file(memory_file, format="mp3")
        play(sound)
