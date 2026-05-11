#!/usr/bin/env python3
"""
Analysis: Can we play WAV audio through ESP32 buzzer (LEDC PWM)?

This script analyzes the feasibility and shows the concept.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_pwm_audio_playback():
    """Analyze PWM audio playback feasibility."""

    print("=" * 80)
    print("Can ESP32 Buzzer (LEDC PWM) Play WAV Audio?")
    print("=" * 80)

    # WAV audio parameters
    sample_rate = 16000  # 16 kHz
    sample_period_us = 1_000_000 / sample_rate  # 62.5 microseconds

    # ESP32 LEDC parameters
    pwm_resolution_bits = 10  # 10-bit PWM (0-1023)
    pwm_max_freq = 80_000_000  # 80 MHz APB clock

    print(f"\n📊 WAV Audio Requirements:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Sample period: {sample_period_us:.2f} µs")
    print(f"  Update rate needed: {sample_rate} updates/second")

    print(f"\n⚡ ESP32 LEDC PWM Capabilities:")
    print(f"  PWM resolution: {pwm_resolution_bits} bits ({2**pwm_resolution_bits} levels)")
    print(f"  APB clock: {pwm_max_freq / 1_000_000:.0f} MHz")
    print(f"  Max PWM frequency: ~{pwm_max_freq / (2**pwm_resolution_bits) / 1000:.1f} kHz")

    # Calculate if CPU can update PWM fast enough
    print(f"\n🔬 Feasibility Analysis:")

    # Method 1: Software updates (ledcWrite in loop)
    cpu_freq = 240_000_000  # 240 MHz CPU
    instructions_per_update = 100  # Rough estimate: read sample, convert, ledcWrite
    time_per_update_us = (instructions_per_update / cpu_freq) * 1_000_000

    print(f"\n  Method 1: Software PWM Updates (ledcWrite in loop)")
    print(f"    CPU frequency: {cpu_freq / 1_000_000:.0f} MHz")
    print(f"    Estimated instructions per update: ~{instructions_per_update}")
    print(f"    Time per update: ~{time_per_update_us:.2f} µs")
    print(f"    Time available per sample: {sample_period_us:.2f} µs")

    if time_per_update_us < sample_period_us:
        print(f"    ✅ FEASIBLE: {time_per_update_us:.2f} µs < {sample_period_us:.2f} µs")
        print(f"    CPU usage: ~{(time_per_update_us / sample_period_us) * 100:.1f}%")
    else:
        print(f"    ❌ NOT FEASIBLE: Too slow!")

    # Method 2: DMA (not available for LEDC)
    print(f"\n  Method 2: DMA-based updates")
    print(f"    ❌ NOT AVAILABLE: LEDC doesn't support DMA")
    print(f"    (This is why I2S is used for audio - it has DMA)")

    # Audio quality analysis
    print(f"\n🎵 Expected Audio Quality:")

    # PWM filtering
    print(f"\n  PWM to Analog Conversion:")
    print(f"    PWM frequency needed: >{sample_rate * 10 / 1000:.0f} kHz (10x sample rate)")
    pwm_freq_at_10bit = pwm_max_freq / (2**pwm_resolution_bits)
    print(f"    PWM frequency at 10-bit: {pwm_freq_at_10bit / 1000:.1f} kHz")

    if pwm_freq_at_10bit > sample_rate * 10:
        print(f"    ✅ Sufficient for basic filtering")
    else:
        print(f"    ⚠️  May have PWM noise")

    # Resolution comparison
    print(f"\n  Resolution Comparison:")
    print(f"    WAV audio: 16-bit = 65,536 levels")
    print(f"    PWM output: {pwm_resolution_bits}-bit = {2**pwm_resolution_bits} levels")
    print(f"    Loss: {16 - pwm_resolution_bits} bits = {2**(16-pwm_resolution_bits)}x compression")
    print(f"    Quality: {'Poor' if pwm_resolution_bits < 8 else 'Acceptable'}")

    # Show sample conversion
    print(f"\n📝 Sample Conversion Example:")
    wav_sample_16bit = -5461  # From your debug output
    wav_sample_normalized = (wav_sample_16bit + 32768) / 65536  # 0.0 to 1.0
    pwm_duty = int(wav_sample_normalized * (2**pwm_resolution_bits))

    print(f"  WAV sample (16-bit signed): {wav_sample_16bit}")
    print(f"  Normalized (0.0-1.0): {wav_sample_normalized:.4f}")
    print(f"  PWM duty cycle ({pwm_resolution_bits}-bit): {pwm_duty} / {2**pwm_resolution_bits}")
    print(f"  ledcWrite(channel, {pwm_duty});")

    # Hardware requirements
    print(f"\n🔌 Hardware Requirements:")
    print(f"  Low-pass filter needed:")
    print(f"    RC filter: 1kΩ resistor + 10µF capacitor")
    print(f"    Cutoff frequency: ~{1/(2*3.14159*1000*0.00001)/1000:.1f} kHz")
    print(f"    Purpose: Remove PWM carrier frequency, keep audio")

    # Conclusion
    print(f"\n" + "=" * 80)
    print(f"CONCLUSION:")
    print(f"=" * 80)
    print(f"✅ TECHNICALLY POSSIBLE but with MAJOR limitations:")
    print(f"   1. ⚠️  High CPU usage (~{(time_per_update_us / sample_period_us) * 100:.1f}%) - blocks other tasks")
    print(f"   2. ⚠️  Lower quality ({pwm_resolution_bits}-bit vs 16-bit)")
    print(f"   3. ⚠️  Requires precise timing (timer interrupt)")
    print(f"   4. ⚠️  Needs external RC filter")
    print(f"   5. ⚠️  No DMA - all software based")
    print(f"\n❌ NOT RECOMMENDED because:")
    print(f"   - I2S DAC already available on GPIO 25")
    print(f"   - I2S has DMA (no CPU overhead)")
    print(f"   - I2S provides better quality")
    print(f"   - AudioOutputI2S library already implemented")
    print(f"\n💡 RECOMMENDATION: Use AudioOutputI2S (current implementation)")
    print(f"=" * 80)


if __name__ == "__main__":
    analyze_pwm_audio_playback()
