"""
berg_lyrical.py — Alban Berg-inspired lyrical serial synthesis

The anti-Webern. Where Webern was sparse and pointillist,
Berg was lush, romantic, and dramatic. His serialism always
reached back toward tonality.

Characteristics modeled from Berg's mature works (Violin Concerto,
Lyric Suite, Wozzeck):
  - Tonal row: 12-tone row that outlines triads + whole-tone tail
    (modeled on the Violin Concerto row concept)
  - Long, overlapping legato lines — voices weave and breathe
  - Dense layered textures: 3-5 simultaneous voices
  - Wide dynamic range: pp through ff (Berg was not afraid of loud)
  - Tonal "windows": moments where the harmony briefly resolves
  - Waltz fragments: rhythmic groupings in 3 (Berg's obsession)
  - Expressionist swells: crescendo/diminuendo within phrases
  - Duration: 90 seconds — Berg was never brief

Usage:
    python berg_lyrical.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import random
import argparse

from webern_pointillism import (
    TIMBRES, SAMPLE_RATE, pointillist_tone, simple_reverb,
    freq_from_pitch_class
)

DURATION = 90  # Berg needs room to breathe

# =====================================================================
# ROW: Berg's Violin Concerto concept — triads built into serial order
# G Bb D F# A C E G# B C# D# F
# Outlines: Gm, D, Am, E — then whole-tone ascent
# We transpose to start on 0 (C) for simplicity:
# =====================================================================
BERG_ROW = np.array([7, 10, 2, 6, 9, 0, 4, 8, 11, 1, 3, 5])
# This gives us: G Bb D F# A C E G# B C# D# F
# Triads visible: [G,Bb,D]=Gm  [D,F#,A]=D  [A,C,E]=Am  [E,G#,B]=E
# Whole-tone tail: [G#,B,C#,D#,F] ≈ whole-tone fragment

def row_inversion(row):
    return (2 * row[0] - row) % 12

def row_retrograde(row):
    return row[::-1]

def row_transpose(row, interval):
    return (row + interval) % 12


def make_legato_line(rng, row_form, start_time, duration, octave_center,
                     amp_base, amp_peak, timbre_pool, waltz=False):
    """Generate a single long melodic line — Berg's lyrical voice.

    Unlike Webern's isolated points, Berg's voices are continuous
    legato phrases with overlapping tones and expressive swells.
    """
    events = []
    cursor = start_time

    # Walk through the row, possibly repeating
    pitches = list(row_form)
    # Berg often cycled through row forms freely
    while cursor < start_time + duration:
        for pc in pitches:
            if cursor >= start_time + duration:
                break

            # Tone duration: long and overlapping
            if waltz:
                # Waltz grouping: alternating long-short-short
                beat_in_group = len(events) % 3
                if beat_in_group == 0:
                    tone_dur = rng.uniform(2.5, 4.0)   # downbeat: long
                else:
                    tone_dur = rng.uniform(1.0, 2.0)    # upbeats: shorter
            else:
                tone_dur = rng.uniform(2.0, 5.0)  # free legato

            # Register: stays close to center, occasional leap
            if rng.random() < 0.15:
                octave = octave_center + rng.choice([-2, -1, 1, 2])
            else:
                octave = octave_center + rng.choice([-1, 0, 0, 0, 1])
            octave = max(2, min(6, octave))

            # Dynamic arc within the phrase — expressionist swell
            phrase_pos = (cursor - start_time) / max(duration, 1)
            # Bell curve: swell to middle, fade at edges
            arc = np.sin(phrase_pos * np.pi)
            amp = amp_base + (amp_peak - amp_base) * arc
            # Add some local variation
            amp *= rng.uniform(0.85, 1.15)

            timbre = rng.choice(timbre_pool)

            events.append({
                "time": cursor,
                "pc": int(pc),
                "octave": octave,
                "duration": tone_dur,
                "amplitude": min(amp, 0.35),
                "timbre": timbre
            })

            # Overlap: next note starts before this one ends
            advance = tone_dur * rng.uniform(0.4, 0.8)
            cursor += advance

    return events


def generate_berg_piece(seed=None):
    """Generate a lush, layered piece using Berg-derived techniques."""

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)

    row_P = BERG_ROW
    row_I = row_inversion(row_P)
    row_R = row_retrograde(row_P)
    row_RI = row_retrograde(row_I)

    # Timbre groupings — Berg's orchestration is stratified
    strings = [TIMBRES[0], TIMBRES[1]]  # cello pont, cello tasto
    winds = [TIMBRES[2], TIMBRES[3], TIMBRES[7]]  # flute, clarinet, oboe
    colors = [TIMBRES[4], TIMBRES[5]]  # bell, glass

    all_events = []

    # ================================================================
    # SECTION 1: Opening — two string voices, pp, slowly building
    # ================================================================
    print("  Section 1: Opening (0-30s) — two string voices emerging")

    # Voice 1: primary row, low register, cello tasto
    events_v1 = make_legato_line(
        rng, row_P, start_time=0.5, duration=28,
        octave_center=3, amp_base=0.06, amp_peak=0.15,
        timbre_pool=strings
    )
    all_events.extend(events_v1)

    # Voice 2: inverted row, higher, enters at 4s (staggered like Berg)
    events_v2 = make_legato_line(
        rng, row_I, start_time=4.0, duration=24,
        octave_center=4, amp_base=0.04, amp_peak=0.12,
        timbre_pool=strings
    )
    all_events.extend(events_v2)

    # ================================================================
    # SECTION 2: Development — winds join, waltz fragments, crescendo
    # ================================================================
    print("  Section 2: Development (25-65s) — winds, waltz, crescendo")

    # Voice 3: retrograde, wind color, waltz rhythm
    events_v3 = make_legato_line(
        rng, row_R, start_time=25.0, duration=38,
        octave_center=4, amp_base=0.08, amp_peak=0.25,
        timbre_pool=winds, waltz=True
    )
    all_events.extend(events_v3)

    # Voice 4: transposed prime, strings, building underneath
    events_v4 = make_legato_line(
        rng, row_transpose(row_P, 5), start_time=28.0, duration=35,
        octave_center=3, amp_base=0.07, amp_peak=0.20,
        timbre_pool=strings
    )
    all_events.extend(events_v4)

    # Voice 5: RI form, high register, wind — the dramatic soprano
    events_v5 = make_legato_line(
        rng, row_RI, start_time=35.0, duration=25,
        octave_center=5, amp_base=0.05, amp_peak=0.22,
        timbre_pool=winds
    )
    all_events.extend(events_v5)

    # ================================================================
    # SECTION 3: Tonal window — brief moment of near-resolution
    # ================================================================
    print("  Section 3: Tonal window (58-72s) — G minor triad emerges")

    # Extract the triadic fragment from the row: G Bb D (positions 0-2)
    triad_pcs = [7, 10, 2]  # G, Bb, D = G minor
    for i, pc in enumerate(triad_pcs):
        start = 58.0 + i * 1.5
        timbre = rng.choice(strings)
        all_events.append({
            "time": start,
            "pc": pc,
            "octave": 3,
            "duration": rng.uniform(6.0, 9.0),
            "amplitude": 0.14,
            "timbre": timbre
        })
        # Double at octave for richness
        all_events.append({
            "time": start + 0.3,
            "pc": pc,
            "octave": 4,
            "duration": rng.uniform(5.0, 7.0),
            "amplitude": 0.08,
            "timbre": rng.choice(winds)
        })

    # Color tones — bell/glass for the "window" shimmer
    for _ in range(4):
        all_events.append({
            "time": rng.uniform(60, 70),
            "pc": rng.choice(triad_pcs),
            "octave": rng.choice([5, 6]),
            "duration": rng.uniform(3.0, 6.0),
            "amplitude": rng.uniform(0.03, 0.07),
            "timbre": rng.choice(colors)
        })

    # ================================================================
    # SECTION 4: Dissolution — fading back into chromaticism
    # ================================================================
    print("  Section 4: Dissolution (70-88s) — fading, fragmenting")

    # Single voice, retrograde, slowing down, getting quieter
    events_final = make_legato_line(
        rng, row_R, start_time=70.0, duration=17,
        octave_center=3, amp_base=0.03, amp_peak=0.08,
        timbre_pool=[TIMBRES[1], TIMBRES[5]]  # tasto + glass = dark + ethereal
    )
    all_events.extend(events_final)

    # A few isolated color tones — the piece evaporating
    for i in range(3):
        all_events.append({
            "time": 80.0 + i * 2.5,
            "pc": int(row_P[i]),
            "octave": rng.choice([5, 6]),
            "duration": rng.uniform(2.0, 4.0),
            "amplitude": rng.uniform(0.02, 0.04),
            "timbre": rng.choice(colors)
        })

    # ================================================================
    # RENDER
    # ================================================================
    print(f"\n  Rendering {len(all_events)} events...")

    for event in all_events:
        audio += pointillist_tone(
            t, event["time"], event["pc"], event["octave"],
            event["duration"], event["amplitude"], event["timbre"],
            np_rng
        )

    # Reverb — wetter than Webern, Berg needs a concert hall
    audio = simple_reverb(audio, decay=0.5, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 1.5, 1.0)
    fade_out = np.minimum((DURATION - t) / 3.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.1)

    audio = np.tanh(audio * 1.1) / 1.1

    return audio, all_events


def main():
    parser = argparse.ArgumentParser(description="Berg-inspired lyrical serial synthesis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="berg_01.wav", help="Output filename")
    args = parser.parse_args()

    print(f"Generating Berg lyrical piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Row: Violin Concerto-derived (triad-outlining)\n")

    audio, events = generate_berg_piece(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")
    print(f"  Sections: Opening → Development → Tonal window → Dissolution")


if __name__ == "__main__":
    main()
