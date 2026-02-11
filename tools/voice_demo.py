"""
voice_demo.py — Vocal synthesis test bench

Isolated sustained vocal tones so you can hear what the
formant synthesis is actually doing without any composition.

Three passes:
  1. Voice gallery: each voice type (soprano/alto/tenor/bass) on
     a comfortable pitch, ~6 seconds each, vowel morphing audible
  2. Vowel sweep: single voice (alto) cycling through all 6 vowels
     on one sustained pitch — hear the formant shapes change
  3. Choir chord: all four voices on a G minor triad, sustained,
     to hear how they blend

Usage:
    python voice_demo.py [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import argparse

from webern_pointillism import SAMPLE_RATE, simple_reverb
from berg_vocal import (
    VOICE_TIMBRES, VOWELS, vocal_tone
)

TONE_DUR = 7.0   # seconds per demo tone
GAP = 2.5        # silence between tones
DEMO_DURATION = 180


def main():
    parser = argparse.ArgumentParser(description="Vocal synthesis demo")
    parser.add_argument("--output", type=str, default="voice_demo.wav",
                        help="Output filename")
    args = parser.parse_args()

    np_rng = np.random.RandomState(42)
    t = np.linspace(0, DEMO_DURATION, int(SAMPLE_RATE * DEMO_DURATION), endpoint=False)
    audio = np.zeros_like(t)

    cursor = 1.5

    # ---- PASS 1: Voice type gallery ----
    print("=== Voice Gallery ===\n")

    # Each voice at a comfortable pitch in their range
    voice_demos = [
        (VOICE_TIMBRES[3], 7, 2, "Bass — G2 (98 Hz)"),
        (VOICE_TIMBRES[2], 9, 3, "Tenor — A3 (220 Hz)"),
        (VOICE_TIMBRES[1], 0, 4, "Alto — C4 (262 Hz)"),
        (VOICE_TIMBRES[0], 4, 5, "Soprano — E5 (659 Hz)"),
    ]

    for vtimbre, pc, octave, label in voice_demos:
        print(f"  {cursor:6.1f}s - {cursor + TONE_DUR:5.1f}s  {label}  vowels: {' → '.join(vtimbre['vowel_sequence'])}")

        audio += vocal_tone(
            t, start=cursor,
            pitch_class=pc, octave=octave,
            duration=TONE_DUR, amplitude=0.28,
            voice_timbre=vtimbre, rng=np_rng
        )
        cursor += TONE_DUR + GAP

    # ---- PASS 2: Vowel sweep on single voice ----
    cursor += 2.0
    print(f"\n=== Vowel Sweep (Alto, C4) ===\n")

    # Create a custom voice timbre for each vowel (held, not morphing)
    vowel_names = ["ah", "ee", "oh", "oo", "eh", "mm"]
    alto_base = VOICE_TIMBRES[1].copy()

    for vowel_name in vowel_names:
        # Override the vowel sequence to hold a single vowel
        held_timbre = alto_base.copy()
        held_timbre["vowel_sequence"] = [vowel_name, vowel_name]
        held_timbre["name"] = f"alto_{vowel_name}"

        print(f"  {cursor:6.1f}s - {cursor + TONE_DUR:5.1f}s  \"{vowel_name}\"")

        audio += vocal_tone(
            t, start=cursor,
            pitch_class=0, octave=4,  # C4
            duration=TONE_DUR, amplitude=0.28,
            voice_timbre=held_timbre, rng=np_rng
        )
        cursor += TONE_DUR + GAP

    # ---- PASS 3: Choir chord — G minor (G-Bb-D-G) ----
    cursor += 3.0
    chord_dur = 12.0
    print(f"\n=== Choir Chord: G minor ({cursor:.0f}s - {cursor + chord_dur:.0f}s) ===\n")

    choir_parts = [
        (VOICE_TIMBRES[3], 7, 2, "Bass — G2"),      # G
        (VOICE_TIMBRES[2], 10, 3, "Tenor — Bb3"),    # Bb
        (VOICE_TIMBRES[1], 2, 4, "Alto — D4"),       # D
        (VOICE_TIMBRES[0], 7, 5, "Soprano — G5"),    # G
    ]

    for vtimbre, pc, octave, label in choir_parts:
        # Stagger entries slightly for realism
        offset = choir_parts.index((vtimbre, pc, octave, label)) * 0.4
        print(f"  {cursor + offset:6.1f}s  {label}")

        audio += vocal_tone(
            t, start=cursor + offset,
            pitch_class=pc, octave=octave,
            duration=chord_dur - offset, amplitude=0.20,
            voice_timbre=vtimbre, rng=np_rng
        )

    cursor += chord_dur + GAP

    # ---- PASS 4: Unison — all four voices on same pitch (A3) ----
    cursor += 2.0
    unison_dur = 10.0
    print(f"\n=== Unison A3 — all four voice types ({cursor:.0f}s - {cursor + unison_dur:.0f}s) ===\n")

    for vtimbre in VOICE_TIMBRES:
        # Adjust octave ranges to all hit A3
        print(f"  {cursor:6.1f}s  {vtimbre['name']} on A3")
        audio += vocal_tone(
            t, start=cursor,
            pitch_class=9, octave=3,
            duration=unison_dur, amplitude=0.18,
            voice_timbre=vtimbre, rng=np_rng
        )

    cursor += unison_dur

    # Trim
    actual_samples = min(int((cursor + 3.0) * SAMPLE_RATE), len(audio))
    audio = audio[:actual_samples]
    t_trim = t[:actual_samples]

    # Reverb — moderate
    audio = simple_reverb(audio, decay=0.35, sample_rate=SAMPLE_RATE)

    # Gentle fades
    fade_in = np.minimum(t_trim / 0.5, 1.0)
    fade_out = np.minimum((t_trim[-1] - t_trim) / 2.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.08)

    audio = np.tanh(audio * 1.02) / 1.02

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    actual_dur = actual_samples / SAMPLE_RATE
    print(f"\nGenerated: {args.output}")
    print(f"  Duration: {actual_dur:.0f}s")
    print(f"  Sections: Voice gallery → Vowel sweep → Choir chord → Unison")


if __name__ == "__main__":
    main()
