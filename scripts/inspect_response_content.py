#!/usr/bin/env python3
"""
Script to inspect and understand response.content from the random endpoint.
This script provides detailed information about the audio data being returned.
"""

import io
import requests
from pydub import AudioSegment
from pydub.playback import play


def inspect_response_content():
    """Fetch and analyze the response from the random endpoint."""

    print("=" * 60)
    print("Fetching response from http://127.0.0.1:7777/random")
    print("=" * 60)

    # Send POST request to random generator of text
    response = requests.post("http://127.0.0.1:7777/random")

    # Basic response information
    print(f"\n📊 Response Status Code: {response.status_code}")
    print(f"📋 Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
    print(f"📦 Content-Length: {len(response.content)} bytes")

    # Headers
    print("\n🔍 Response Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")

    # Content analysis
    print("\n🎵 Content Analysis:")
    print(f"  First 100 bytes (hex): {response.content[:100].hex()}")
    print(f"  First 50 bytes (repr): {repr(response.content[:50])}")

    # Try to identify WAV header
    if response.content[:4] == b'RIFF':
        print("  ✅ Detected RIFF/WAV header")
        print(f"  File size in header: {int.from_bytes(response.content[4:8], 'little')} bytes")
        print(f"  Format: {response.content[8:12]}")

        # Parse fmt chunk
        fmt_pos = response.content.find(b'fmt ')
        if fmt_pos != -1:
            print(f"\n  📝 fmt chunk at position {fmt_pos}:")
            chunk_size = int.from_bytes(response.content[fmt_pos+4:fmt_pos+8], 'little')
            print(f"    Chunk size: {chunk_size} bytes")
            audio_format = int.from_bytes(response.content[fmt_pos+8:fmt_pos+10], 'little')
            print(f"    Audio format: {audio_format} (1=PCM)")
            num_channels = int.from_bytes(response.content[fmt_pos+10:fmt_pos+12], 'little')
            print(f"    Channels: {num_channels}")
            sample_rate = int.from_bytes(response.content[fmt_pos+12:fmt_pos+16], 'little')
            print(f"    Sample rate: {sample_rate} Hz")
            byte_rate = int.from_bytes(response.content[fmt_pos+16:fmt_pos+20], 'little')
            print(f"    Byte rate: {byte_rate} bytes/sec")
            block_align = int.from_bytes(response.content[fmt_pos+20:fmt_pos+22], 'little')
            print(f"    Block align: {block_align} bytes")
            bits_per_sample = int.from_bytes(response.content[fmt_pos+22:fmt_pos+24], 'little')
            print(f"    Bits per sample: {bits_per_sample}")

        # Parse data chunk
        data_pos = response.content.find(b'data')
        if data_pos != -1:
            print(f"\n  🎵 data chunk at position {data_pos}:")
            data_size = int.from_bytes(response.content[data_pos+4:data_pos+8], 'little')
            print(f"    Data chunk size: {data_size} bytes")
            data_start = data_pos + 8
            print(f"    Audio data starts at byte: {data_start}")
            print(f"    First 20 bytes of audio data (hex): {response.content[data_start:data_start+20].hex()}")
            print(f"    First 10 samples (16-bit signed):")
            for i in range(min(10, data_size//2)):
                sample_pos = data_start + i * 2
                sample = int.from_bytes(response.content[sample_pos:sample_pos+2], 'little', signed=True)
                print(f"      Sample {i}: {sample}")
                print(f"      Sample {i}: {response.content[sample_pos:sample_pos+2]}")

            for i in range(min(100, data_size//2)):
                sample_pos = data_start + 20000 + i * 2
                sample = int.from_bytes(response.content[sample_pos:sample_pos+2], 'little', signed=True)
                print(f"      Sample {i+20000}: {sample}")
                print(f"      Sample {i+20000}: {response.content[sample_pos:sample_pos+2]}")
    else:
        print("  ⚠️  No RIFF header detected - may not be WAV format")

    # Load into pydub for audio analysis
    print("\n🎧 Audio Properties:")
    try:
        memory_file = io.BytesIO(response.content)
        sound = AudioSegment.from_file(memory_file, format="wav")

        print(f"  Duration: {len(sound)} ms ({len(sound)/1000:.2f} seconds)")
        print(f"  Channels: {sound.channels}")
        print(f"  Sample Width: {sound.sample_width} bytes ({sound.sample_width * 8} bit)")
        print(f"  Frame Rate: {sound.frame_rate} Hz")
        print(f"  Frame Width: {sound.frame_width} bytes")
        print(f"  Max dBFS: {sound.max_dBFS:.2f} dB")
        print(f"  RMS: {sound.rms:.2f}")

        # Ask if user wants to play
        play(sound)
        # print("\n" + "=" * 60)
        # user_input = input("▶️  Play the audio? (y/n): ").strip().lower()
        # if user_input == 'y':
        #     print("Playing audio...")
        #     play(sound)
        #     print("✅ Playback complete")
        # else:
        #     print("⏭️  Skipped playback")

        # # Ask if user wants to save
        # print("\n" + "=" * 60)
        # save_input = input("💾 Save audio to file? (y/n): ").strip().lower()
        # if save_input == 'y':
        #     filename = input("Enter filename (default: response_audio.wav): ").strip()
        #     if not filename:
        #         filename = "response_audio.wav"
        #     if not filename.endswith('.wav'):
        #         filename += '.wav'

        #     with open(filename, 'wb') as f:
        #         f.write(response.content)
        #     print(f"✅ Saved to {filename}")

    except Exception as e:
        print(f"  ❌ Error loading audio: {e}")

    print("\n" + "=" * 60)
    print("Inspection complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        inspect_response_content()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to http://127.0.0.1:7777/random")
        print("   Make sure the server is running!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
