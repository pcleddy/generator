"""
microtonal_descent.py — Bizarre microtonal descent.

Slow, meditative piece: single notes and pairs of nylon guitar,
drifting downward by microtones from a half step to a minor 3rd
over 30 seconds. The intervals between paired notes warp as
each voice drifts at a slightly different rate, turning perfect
fifths into wolves, octaves into something alien.

Direct synthesis — bypasses Renderer for exact frequency control.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile
import subprocess

from synthesis_engine.config import SAMPLE_RATE
from synthesis_engine.seed_manager import SeedManager
from synthesis_engine.synthesis.plucked import karplus_strong
from synthesis_engine.synthesis.reverb import simple_reverb
from synthesis_engine.profiles.plucked import PLUCKED_PROFILES

PROFILE = PLUCKED_PROFILES["guitar_nylon"]


def cents_to_ratio(cents):
    return 2 ** (cents / 1200.0)


def generate():
    rng = SeedManager(77)

    duration = 34.0  # extra tail room
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    audio = np.zeros(len(t))

    # Starting pitch: B3 (246.94 Hz) — rich guitar register
    base_freq = 246.94

    # Total drift: -300 cents (minor 3rd down) over ~28 seconds of content
    # But each voice in a pair drifts at a slightly different rate
    # Voice 1 (lower): drifts the full -300 cents
    # Voice 2 (upper): drifts -240 cents (less) — so intervals WARP

    # Event sequence: time, type, intervals (semitones above base for each voice)
    # "single" = one note, "pair" = two notes
    events = [
        # Opening: single B3, pure, unhurried
        (0.5,   "single", [0]),

        # First pair: perfect 5th (B3 + F#4) — still mostly pure
        (3.8,   "pair",   [0, 7]),

        # Single low note
        (7.0,   "single", [-5]),

        # Pair: octave — starting to bend
        (9.5,   "pair",   [0, 12]),

        # Single, reaching up
        (12.5,  "single", [5]),

        # Pair: tritone interval — deliberately unsettling
        (15.0,  "pair",   [0, 6]),

        # Single low murmur
        (17.8,  "single", [-3]),

        # Pair: minor 2nd cluster — dissonant by now with drift
        (20.0,  "pair",   [0, 1]),

        # Single note, high and lonely
        (22.5,  "single", [10]),

        # Final pair: what was a perfect 5th is now deeply warped
        (25.0,  "pair",   [0, 7]),

        # Last breath: single low note, maximally drifted
        (27.5,  "single", [0]),
    ]

    total_content_time = 28.0  # seconds of musical content
    drift_per_second = -300.0 / total_content_time  # cents per second

    # Voice 2 drifts less — intervals widen/narrow unpredictably
    drift_per_second_v2 = -220.0 / total_content_time

    for event_time, event_type, intervals in events:
        # Calculate drift at this moment
        drift_v1 = drift_per_second * event_time  # cents, negative = down
        drift_v2 = drift_per_second_v2 * event_time

        for i, semitones in enumerate(intervals):
            # Base frequency shifted by interval
            freq = base_freq * (2 ** (semitones / 12.0))

            # Apply microtonal drift (voice 1 or voice 2 rate)
            drift = drift_v1 if i == 0 else drift_v2
            freq *= cents_to_ratio(drift)

            # Tiny humanization
            freq *= cents_to_ratio(rng.uniform(-3, 3))

            # Amplitude: gentle, consistent, pairs slightly softer per voice
            amp = 0.55 if event_type == "single" else 0.45
            amp *= rng.uniform(0.92, 1.08)

            # Long note duration — let it ring
            note_dur = rng.uniform(2.5, 3.5)

            audio += karplus_strong(
                t, event_time, freq, note_dur, min(amp, 0.6),
                PROFILE, rng
            )

        # Print progress
        print(f"  t={event_time:5.1f}s  {event_type:6s}  "
              f"drift_v1={drift_v1:+.0f}¢  drift_v2={drift_v2:+.0f}¢")

    # Generous reverb — cathedral-like for the slow pace
    audio = simple_reverb(audio, preset="cathedral", sample_rate=SAMPLE_RATE)

    # Long fade out
    fade_len = int(3.0 * SAMPLE_RATE)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.90

    total_drift_v1 = drift_per_second * total_content_time
    total_drift_v2 = drift_per_second_v2 * total_content_time
    print(f"\n  Voice 1 total drift: {total_drift_v1:+.0f} cents "
          f"({abs(total_drift_v1)/100:.1f} semitones down)")
    print(f"  Voice 2 total drift: {total_drift_v2:+.0f} cents "
          f"({abs(total_drift_v2)/100:.1f} semitones down)")
    print(f"  Interval warping: {abs(total_drift_v1 - total_drift_v2):.0f} cents "
          f"of divergence between voices")

    return audio, duration


def main():
    print("Microtonal Descent — Bizarre Downward Drift")
    print("  B3 → drifting down 300 cents (minor 3rd)")
    print("  Paired intervals warp as voices drift at different rates")
    print()

    audio, dur = generate()

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "microtonal_descent.wav")
    mp3_path = os.path.join(out_dir, "audio", "microtonal_descent.mp3")

    audio_16 = np.int16(audio * 32767)
    wavfile.write(wav_path, SAMPLE_RATE, audio_16)
    print(f"\n  Written: {wav_path}")

    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-codec:a', 'libmp3lame', '-b:a', '192k', mp3_path],
        capture_output=True
    )
    print(f"  Written: {mp3_path}")


if __name__ == "__main__":
    main()
