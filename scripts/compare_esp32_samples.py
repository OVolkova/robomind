#!/usr/bin/env python3
"""
Compare ESP32 buffer samples with actual server response.
Helps verify data integrity through the streaming pipeline.
"""

import requests
import io
from pydub import AudioSegment


def hex_to_samples(hex_str):
    """Convert hex string from ESP32 debug output to sample values."""
    hex_bytes = bytes.fromhex(hex_str.replace(' ', ''))
    samples = []
    for i in range(0, len(hex_bytes), 2):
        if i + 1 < len(hex_bytes):
            # Little-endian 16-bit signed integer
            sample = int.from_bytes(hex_bytes[i:i+2], 'little', signed=True)
            samples.append(sample)
    return samples


def compare_esp32_buffer_with_server():
    """Fetch server response and compare with ESP32 buffer data."""

    print("=" * 80)
    print("Comparing ESP32 Buffer Data with Server Response")
    print("=" * 80)

    # Fetch from server
    print("\nFetching response from http://127.0.0.1:7777/random...")
    response = requests.post("http://127.0.0.1:7777/random")

    # Parse WAV data
    data_pos = response.content.find(b'data')
    if data_pos == -1:
        print("Error: Could not find data chunk")
        return

    data_size = int.from_bytes(response.content[data_pos+4:data_pos+8], 'little')
    data_start = data_pos + 8

    print(f"WAV data chunk size: {data_size} bytes")
    print(f"Audio data starts at byte: {data_start}")

    # Example ESP32 buffer hex from your debug output
    # Update these with actual values from your ESP32 serial output
    esp32_first_20_hex = "AB EA CA E8 AD E5 A8 E3 4C EC 01 FC D9 0A CC 12 26 17 54 17"
    esp32_last_20_hex = "DE 0C 30 0D B8 0E A0 09 06 01 D9 F9 AB F7 FB F7 94 FA 41 FE"

    print("\n" + "=" * 80)
    print("ESP32 Buffer Data (from serial output):")
    print("=" * 80)
    print(f"First 20 bytes: {esp32_first_20_hex}")
    print(f"Last 20 bytes:  {esp32_last_20_hex}")

    # Convert to samples
    esp32_first_samples = hex_to_samples(esp32_first_20_hex)
    esp32_last_samples = hex_to_samples(esp32_last_20_hex)

    print(f"\nFirst 10 samples from ESP32: {esp32_first_samples}")
    print(f"Last 10 samples from ESP32:  {esp32_last_samples}")

    # Get server data at various positions
    print("\n" + "=" * 80)
    print("Server Response Data:")
    print("=" * 80)

    # First 20 bytes from server
    server_first_20 = response.content[data_start:data_start+20]
    print(f"First 20 bytes: {server_first_20.hex(' ').upper()}")

    # Last 20 bytes from server
    server_last_20 = response.content[data_start+data_size-20:data_start+data_size]
    print(f"Last 20 bytes:  {server_last_20.hex(' ').upper()}")

    # Convert to samples
    server_first_samples = []
    for i in range(0, 20, 2):
        sample = int.from_bytes(server_first_20[i:i+2], 'little', signed=True)
        server_first_samples.append(sample)

    server_last_samples = []
    for i in range(0, 20, 2):
        sample = int.from_bytes(server_last_20[i:i+2], 'little', signed=True)
        server_last_samples.append(sample)

    print(f"\nFirst 10 samples from server: {server_first_samples}")
    print(f"Last 10 samples from server:  {server_last_samples}")

    # Try to find ESP32 pattern in server data
    print("\n" + "=" * 80)
    print("Searching for ESP32 buffer pattern in server data...")
    print("=" * 80)

    esp32_first_bytes = bytes.fromhex(esp32_first_20_hex.replace(' ', ''))

    # Search in server data (skip WAV header)
    position = response.content.find(esp32_first_bytes, data_start)
    if position != -1:
        offset_from_audio_start = position - data_start
        sample_number = offset_from_audio_start // 2
        time_seconds = sample_number / 16000.0
        print(f"✅ Found ESP32 first buffer pattern in server data!")
        print(f"   Position in file: byte {position}")
        print(f"   Offset from audio start: {offset_from_audio_start} bytes")
        print(f"   Sample number: {sample_number}")
        print(f"   Time position: {time_seconds:.3f} seconds into audio")
    else:
        print("❌ ESP32 buffer pattern not found in server data")
        print("   (This is normal if you're comparing different responses)")

    # Show some samples at different positions for reference
    print("\n" + "=" * 80)
    print("Sample Values at Various Positions in Server Response:")
    print("=" * 80)

    positions = [0, 1000, 5000, 10000, 15000, 20000]
    for pos in positions:
        byte_pos = data_start + pos * 2
        if byte_pos + 2 <= data_start + data_size:
            sample_bytes = response.content[byte_pos:byte_pos+2]
            sample = int.from_bytes(sample_bytes, 'little', signed=True)
            time_sec = pos / 16000.0
            print(f"Sample {pos:5d} (t={time_sec:.3f}s): {sample:6d}  [{sample_bytes.hex(' ').upper()}]")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        compare_esp32_buffer_with_server()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to http://127.0.0.1:7777/random")
        print("   Make sure the server is running!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
