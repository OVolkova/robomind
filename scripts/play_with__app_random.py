import io
import requests

from pydub import AudioSegment
from pydub.playback import play


if __name__ == "__main__":
    # Send POST request to random generator of text
    response = requests.post("http://127.0.0.1:7777/random")
    memory_file = io.BytesIO(response.content)
    sound = AudioSegment.from_file(memory_file, format="mp3")
    play(sound)
