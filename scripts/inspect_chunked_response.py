#!/usr/bin/env python3
"""
Script to inspect chunked transfer encoding from the random endpoint.
Shows how data is received in chunks and displays start/end of each chunk.
"""

import requests
import io
from pydub import AudioSegment
from pydub.playback import play


def inspect_chunked_response():
    """Fetch and analyze chunked response from the random endpoint."""

    print("=" * 80)
    print("Inspecting Chunked Response from http://127.0.0.1:7777/random")
    print("=" * 80)

    # Track chunks received
    chunks = []
    total_bytes = 0
    chunk_count = 0

    def chunk_callback(chunk):
        """Callback to capture each chunk as it arrives."""
        nonlocal chunk_count, total_bytes
        if chunk:
            chunk_count += 1
            chunk_size = len(chunk)
            total_bytes += chunk_size
            chunks.append(chunk)
            print(f"\n📦 Chunk {chunk_count}:")
            print(f"  Size: {chunk_size} bytes")
            print(f"  Total received so far: {total_bytes} bytes")
            print(f"  First 20 bytes (hex): {chunk[:20].hex()}")
            print(f"  First 20 bytes (repr): {repr(chunk[:20])}")
            print(f"  Last 20 bytes (hex): {chunk[-20:].hex()}")
            print(f"  Last 20 bytes (repr): {repr(chunk[-20:])}")

    # Send POST request with streaming enabled
    print("\n🚀 Sending POST request with streaming enabled...\n")
    response = requests.post("http://127.0.0.1:7777/random", stream=True)

    # Check if response is chunked
    transfer_encoding = response.headers.get('Transfer-Encoding', 'Not specified')
    print(f"Transfer-Encoding: {transfer_encoding}")
    print(f"Content-Type: {response.headers.get('Content-Type', 'Not specified')}")

    if 'Content-Length' in response.headers:
        print(f"Content-Length: {response.headers['Content-Length']} bytes")
    else:
        print("Content-Length: Not specified (likely chunked)")

    print("\n" + "=" * 80)
    print("Receiving chunks...")
    print("=" * 80)

    # Iterate through chunks
    for chunk in response.iter_content(chunk_size=None):  # None = use server's chunk size
        chunk_callback(chunk)

    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Total chunks received: {chunk_count}")
    print(f"Total bytes received: {total_bytes}")

    if chunk_count > 0:
        avg_chunk_size = total_bytes / chunk_count
        print(f"Average chunk size: {avg_chunk_size:.2f} bytes")
        print(f"Smallest chunk: {min(len(c) for c in chunks)} bytes")
        print(f"Largest chunk: {max(len(c) for c in chunks)} bytes")

    # Reconstruct full content
    full_content = b''.join(chunks)
    print(f"\nReconstructed content size: {len(full_content)} bytes")

    # Analyze the full WAV file
    print("\n" + "=" * 80)
    print("🎵 Analyzing Complete Audio File")
    print("=" * 80)

    if full_content[:4] == b'RIFF':
        print("✅ Valid RIFF/WAV file")

        # Parse data chunk
        data_pos = full_content.find(b'data')
        if data_pos != -1:
            data_size = int.from_bytes(full_content[data_pos+4:data_pos+8], 'little')
            data_start = data_pos + 8
            print(f"Data chunk size: {data_size} bytes")
            print(f"Audio data starts at byte: {data_start}")

            # Map chunks to audio structure
            print(f"\n📍 Chunk Boundaries in WAV Structure:")
            current_pos = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_start = current_pos
                chunk_end = current_pos + len(chunk)

                # Determine what part of WAV this chunk contains
                if chunk_start < 12:
                    section = "RIFF Header"
                elif chunk_start < data_start:
                    section = "WAV Headers (fmt chunk)"
                elif chunk_start < data_start + data_size:
                    section = "Audio Data"
                else:
                    section = "Beyond audio data"

                print(f"  Chunk {i}: bytes {chunk_start}-{chunk_end} ({section})")
                current_pos = chunk_end

    # Try to play the audio
    print("\n" + "=" * 80)
    try:
        memory_file = io.BytesIO(full_content)
        sound = AudioSegment.from_file(memory_file, format="wav")

        print("🎧 Audio Properties:")
        print(f"  Duration: {len(sound)} ms ({len(sound)/1000:.2f} seconds)")
        print(f"  Sample Rate: {sound.frame_rate} Hz")
        print(f"  Channels: {sound.channels}")
        print(f"  Sample Width: {sound.sample_width * 8} bit")

        print("\n▶️  Playing audio...")
        play(sound)
        print("✅ Playback complete")

    except Exception as e:
        print(f"❌ Error loading/playing audio: {e}")

    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        inspect_chunked_response()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to http://127.0.0.1:7777/random")
        print("   Make sure the server is running!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()