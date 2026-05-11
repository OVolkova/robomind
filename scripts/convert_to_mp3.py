#!/usr/bin/env python3
"""
Batch Audio to MP3 Converter
Converts all audio files in a folder to MP3 format using pydub
"""

import os
from pathlib import Path
from pydub import AudioSegment

# Configuration
INPUT_FOLDER = Path("/Users/olly/Documents/books/a")
OUTPUT_FOLDER = Path("/Users/olly/Documents/books/a/mp3_output")
BITRATE = "192k"  # Options: "128k", "192k", "256k", "320k"
QUALITY = 2  # VBR quality: 0 (best) to 9 (worst), or use BITRATE for CBR

# Supported audio formats
AUDIO_EXTENSIONS = {'.aiff', '.aif', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}


def convert_to_mp3(input_path, output_path):
    """Convert audio file to MP3 using pydub"""
    print(f"Converting: {input_path.name}")

    try:
        # Detect format from file extension
        file_format = input_path.suffix[1:].lower()  # Remove the dot

        # Handle AIFF/AIF
        if file_format in ['aif', 'aiff']:
            file_format = 'aiff'

        # Load audio file
        audio = AudioSegment.from_file(str(input_path), format=file_format)

        # Export as MP3
        audio.export(
            str(output_path),
            format='mp3',
            bitrate=BITRATE,
            parameters=["-q:a", "2"]  # Quality setting for VBR
        )

        print(f"  ✓ Saved: {output_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main conversion pipeline"""
    print("=== Audio to MP3 Batch Converter ===\n")

    # Check if input folder exists
    if not INPUT_FOLDER.exists():
        print(f"Error: Input folder does not exist: {INPUT_FOLDER}")
        return

    # Create output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Quality: VBR q={QUALITY} (or CBR {BITRATE})\n")

    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(INPUT_FOLDER.glob(f"*{ext}"))
        audio_files.extend(INPUT_FOLDER.glob(f"*{ext.upper()}"))

    if not audio_files:
        print(f"No audio files found in {INPUT_FOLDER}")
        print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return

    print(f"Found {len(audio_files)} audio file(s)\n")

    # Convert each file
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        output_name = audio_file.stem + ".mp3"
        output_path = OUTPUT_FOLDER / output_name

        print(f"[{i}/{len(audio_files)}] {audio_file.name}")

        if convert_to_mp3(audio_file, output_path):
            success_count += 1
        print()

    # Summary
    print("=== Complete ===")
    print(f"Successfully converted: {success_count}/{len(audio_files)} files")
    print(f"Output location: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
