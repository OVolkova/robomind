"""
Test script for POST /process.

Records from the Mac microphone, sends audio + current robot action to the
/process endpoint, plays the spoken response, and prints the new action command
that should be forwarded to the Petoi firmware.

Usage:
    python scripts/play_with__app.py [--action wkF] [--rounds 10]
"""

import argparse
import base64
import io

import requests
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

# Input audio is still base64 JSON so the server knows where audio ends and
# current_action begins. Output is streaming WAV with metadata in headers.

SERVER = "http://127.0.0.1:7777"


def run(starting_action: str, rounds: int) -> None:
    microphone = sr.Microphone()
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1

    current_action = starting_action
    print(f"Starting action: {current_action}")
    print("Press Ctrl-C to stop.\n")

    for i in range(rounds):
        print(f"[{i + 1}/{rounds}] Talk...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        print("           ... processing")

        wav_bytes = audio.get_wav_data()

        response = requests.post(
            f"{SERVER}/process",
            json={
                "audio": base64.b64encode(wav_bytes).decode(),
                "current_action": current_action,
            },
            stream=True,
        )

        if not response.ok:
            print(f"           Server error {response.status_code}: {response.text}")
            continue

        response_text = response.headers.get("X-Response-Text", "")
        new_action = response.headers.get("X-New-Action")

        print(f"           Robot says : {response_text!r}")
        print(f"           New action : {new_action or '(none)'}")

        if new_action:
            current_action = new_action
            print(f"           → Send to firmware: {new_action}")

        sound = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
        play(sound)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", default="balance", help="Starting robot action (default: balance)"
    )
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of exchanges (default: 10)"
    )
    args = parser.parse_args()

    try:
        run(starting_action=args.action, rounds=args.rounds)
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {SERVER}. Is the server running?")
    except KeyboardInterrupt:
        print("\nStopped.")
