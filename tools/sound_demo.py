"""
sound_demo.py — Timbre gallery / synthesis test bench

Plays each instrument timbre at a comfortable pitch and duration
so you can actually hear what the synthesis engine is doing.

Each tone is ~5-6 seconds with 2 seconds of silence between.
Two passes:
  1. Solo gallery: each timbre alone on A3 (220 Hz) — good midrange
  2. Register demo: one timbre (cello_pont) across octaves 2-6

Usage:
    python sound_demo.py [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import argparse

# Pull in the synthesis engine
from webern_pointillism import (
    TIMBRES, SAMPLE_RATE, pointillist_tone, simple_reverb
)

DEMO_DURATION = 140  # total seconds — enough for all demos
TONE_DURATION = 5.5  # seconds per demo tone
GAP = 2.0            # silence between tones
DEMO_PITCH = 9       # A (pitch class 9)
DEMO_OCTAVE = 3      # A3 = 220 Hz — comfortable midrange


def main():
    parser = argparse.ArgumentParser(description="Synthesis timbre demo")
    parser.add_argument("--output", type=str, default="sound_demo.wav",
                        help="Output filename")
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    t = np.linspace(0, DEMO_DURATION, int(SAMPLE_RATE * DEMO_DURATION), endpoint=False)
    audio = np.zeros_like(t)

    cursor = 1.0  # start time

    # ---- PASS 1: Solo timbre gallery ----
    print("=== Timbre Gallery (A3 / 220 Hz) ===\n")

    for timbre in TIMBRES:
        print(f"  {cursor:6.1f}s - {cursor + TONE_DURATION:5.1f}s  {timbre['name']}")

        audio += pointillist_tone(
            t, start=cursor,
            pitch_class=DEMO_PITCH, octave=DEMO_OCTAVE,
            duration=TONE_DURATION, amplitude=0.22,
            timbre=timbre, rng=rng
        )
        cursor += TONE_DURATION + GAP

    # ---- PASS 2: Register sweep (cello_pont across octaves) ----
    cursor += 2.0  # extra gap before register demo
    print(f"\n=== Register Sweep (cello_pont, A across octaves) ===\n")

    cello = TIMBRES[0]  # cello_pont
    for octave in [2, 3, 4, 5, 6]:
        freq = 220.0 * (2 ** (octave - 3))
        label = f"A{octave} ({freq:.0f} Hz)"
        print(f"  {cursor:6.1f}s - {cursor + TONE_DURATION:5.1f}s  {label}")

        audio += pointillist_tone(
            t, start=cursor,
            pitch_class=DEMO_PITCH, octave=octave,
            duration=TONE_DURATION, amplitude=0.22,
            timbre=cello, rng=rng
        )
        cursor += TONE_DURATION + GAP

    # Trim to actual used length + tail
    actual_samples = int((cursor + 3.0) * SAMPLE_RATE)
    audio = audio[:actual_samples]
    t_trim = t[:actual_samples]

    # Apply reverb
    audio = simple_reverb(audio, decay=0.35, sample_rate=SAMPLE_RATE)

    # Gentle fade in/out
    fade_in = np.minimum(t_trim / 0.3, 1.0)
    fade_out = np.minimum((t_trim[-1] - t_trim) / 1.5, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.1)

    audio = np.tanh(audio * 1.02) / 1.02

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Duration: {cursor + 3:.0f}s")
    print(f"  Timbres demoed: {len(TIMBRES)}")
    print(f"  Register range: A2-A6")


if __name__ == "__main__":
    main()
