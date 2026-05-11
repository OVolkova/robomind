#!/usr/bin/env python3
"""
Audiobook Generator for Russian Text
Converts a TXT file into chapter-based audio files for iPhone
"""

import torch
import torchaudio
import os
import re
from pathlib import Path
import subprocess

# Configuration
INPUT_DOC = "/Users/olly/Documents/books/non-technical/Эльконин Д.Б. - Психология Игры - 1977.txt"
OUTPUT_DIR = Path("./audiobook_output")
AUDIO_FORMAT = "m4a"  # iPhone-friendly format
SAMPLE_RATE = 48000

# TTS Configuration
LANGUAGE = 'ru'
MODEL_ID = 'v3_1_ru'
SPEAKER = 'eugene'  # or 'aidar', 'baya', 'kseniya', 'xenia'


def read_text_file(file_path):
    """Read text from TXT file"""
    print(f"Reading document: {file_path}")
    file_path = Path(file_path)

    print("Reading TXT file...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Successfully read text file")
    return text


def split_into_chapters(text):
    """Split text into chapters based on Russian chapter markers"""
    # Pattern matches: "Глава пятая", "Глава 1", "ГЛАВА I", "Приложение", etc.
    # Matches: Глава + (number OR roman numeral OR russian word for number) OR Приложение
    chapter_pattern = r'(?:^|\n)((?:Глава\s+(?:\d+|[IVXLCDM]+|первая|вторая|третья|четвертая|пятая|шестая|седьмая|восьмая|девятая|десятая)|Приложение)[^\n]*)'

    chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))

    if chapter_matches:
        chapters = []
        for i, match in enumerate(chapter_matches):
            start = match.start()
            # End is the start of next chapter, or end of text
            end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_content = text[start:end].strip()
            chapters.append(chapter_content)

        print(f"Found {len(chapters)} chapters")
        # Print first few chapter titles for verification
        for i, ch in enumerate(chapters[:3], 1):
            title_line = ch.split('\n')[0]
            print(f"  Chapter {i}: {title_line[:60]}...")
        return chapters
    else:
        # No chapters found, treat entire text as one chapter
        print("Warning: No chapters found. Processing entire text as one file.")
        return [text]


def extract_chapter_title(chapter_text):
    """Extract chapter title from the beginning of chapter text"""
    lines = chapter_text.strip().split('\n')
    # First non-empty line is usually the title
    for line in lines[:5]:
        line = line.strip()
        if line and ('глава' in line.lower() or 'ГЛАВА' in line):
            # Clean up the title for filename
            title = re.sub(r'[^\w\s\-]', '', line)
            title = re.sub(r'\s+', '_', title)
            return title[:50]  # Limit length
    return "Chapter"


def generate_audio(text, output_path, model):
    """Generate audio from text using Silero TTS"""
    print(f"Generating audio: {output_path}")

    # Split long text into chunks if needed (Silero has limits)
    max_chars = 1000
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    audio_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        audio = model.apply_tts(text=chunk,
                               speaker=SPEAKER,
                               sample_rate=SAMPLE_RATE)
        audio_chunks.append(audio)

    # Concatenate all audio chunks
    if len(audio_chunks) > 1:
        full_audio = torch.cat(audio_chunks, dim=0)
    else:
        full_audio = audio_chunks[0]

    # Save as WAV first
    wav_path = output_path.with_suffix('.wav')
    torchaudio.save(str(wav_path), full_audio.unsqueeze(0), SAMPLE_RATE)

    # Convert to M4A for iPhone compatibility
    if AUDIO_FORMAT == "m4a":
        m4a_path = output_path.with_suffix('.m4a')
        convert_to_m4a(wav_path, m4a_path)
        os.remove(wav_path)  # Clean up WAV file
        return m4a_path

    return wav_path


def convert_to_m4a(wav_path, m4a_path):
    """Convert WAV to M4A using ffmpeg"""
    print(f"Converting to M4A: {m4a_path.name}")
    try:
        subprocess.run([
            'ffmpeg', '-i', str(wav_path),
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite output file
            str(m4a_path)
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting to M4A: {e}")
        print("Make sure ffmpeg is installed: brew install ffmpeg")
        raise


def load_tts_model():
    """Load Silero TTS model"""
    print("Loading TTS model...")
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=LANGUAGE,
        speaker=MODEL_ID
    )
    return model


def main():
    """Main audiobook generation pipeline"""
    print("=== Russian Audiobook Generator ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Read the document
    text = read_text_file(INPUT_DOC)
    print(f"Document length: {len(text)} characters\n")

    # Split into chapters
    chapters = split_into_chapters(text)

    # Load TTS model
    model = load_tts_model()
    print()

    # Generate audio for each chapter
    for i, chapter in enumerate(chapters, 1):
        chapter_title = extract_chapter_title(chapter)
        chapter_num = f"{i:02d}"

        output_filename = f"{chapter_num}_{chapter_title}"
        output_path = OUTPUT_DIR / output_filename

        print(f"\n[{i}/{len(chapters)}] Processing: {chapter_title}")
        print(f"  Length: {len(chapter)} characters")

        try:
            audio_path = generate_audio(chapter, output_path, model)
            print(f"  ✓ Saved: {audio_path.name}")
        except Exception as e:
            print(f"  ✗ Error generating audio: {e}")
            continue

    print(f"\n=== Complete ===")
    print(f"Generated {len(chapters)} audio files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
