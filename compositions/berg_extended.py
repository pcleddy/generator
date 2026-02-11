"""
berg_extended.py — Extended Berg piece (~7 minutes) for Tanya

"Cheating with a loop" — but musically, this is just what Berg did:
the same row material returns in different transpositions, orchestrations,
tempi, and densities. Classical variation form meets serial technique.

Structure (7 cycles through the material, each evolving):

  Cycle 1:  Emergence     (0-60s)    pp, 2 voices, strings only
  Cycle 2:  Awakening     (55-120s)  p, 3 voices, winds enter
  Cycle 3:  Waltz         (115-180s) mp, waltz rhythm, full ensemble
  Cycle 4:  Tonal Heart   (175-250s) mf, tonal windows, lush
  Cycle 5:  Storm         (245-310s) f, dense 5 voices, dramatic
  Cycle 6:  Remembrance   (305-370s) mp→pp, echoes of earlier cycles
  Cycle 7:  Dissolution   (365-420s) ppp, single voice, evaporating

Total: ~420 seconds (7 minutes)

Each cycle:
  - Transposes the row by a different interval
  - Uses different timbre combinations
  - Varies density, dynamics, and rhythm
  - Cross-fades with adjacent cycles (~5s overlap)

The "loop" isn't a literal repeat — it's the same *process* generating
different *content* each time. Like looking at the same landscape
through different seasons.

Usage:
    python berg_extended.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import random
import argparse

from webern_pointillism import (
    TIMBRES, SAMPLE_RATE, pointillist_tone, simple_reverb,
    freq_from_pitch_class
)

DURATION = 420  # 7 minutes

# Berg's triadic row (same as berg_lyrical.py)
BERG_ROW = np.array([7, 10, 2, 6, 9, 0, 4, 8, 11, 1, 3, 5])

def row_inversion(row):
    return (2 * row[0] - row) % 12

def row_retrograde(row):
    return row[::-1]

def row_transpose(row, interval):
    return (row + interval) % 12


# Timbre groupings
STRINGS = [TIMBRES[0], TIMBRES[1]]           # cello pont, cello tasto
WINDS = [TIMBRES[2], TIMBRES[3], TIMBRES[7]] # flute, clarinet, oboe
COLORS = [TIMBRES[4], TIMBRES[5]]            # bell, glass
PLUCKED = [TIMBRES[6]]                        # pizzicato
ALL_TIMBRES = STRINGS + WINDS + COLORS


def make_legato_line(rng, row_form, start_time, duration, octave_center,
                     amp_base, amp_peak, timbre_pool, waltz=False,
                     tempo_mult=1.0, arc_shape="bell"):
    """Generate a single long melodic line with configurable arc shape.

    arc_shape:
      "bell"     — swell to middle, fade at edges (standard Berg)
      "rise"     — gradual crescendo
      "fall"     — gradual diminuendo
      "plateau"  — quick rise, sustain, quick fall
    """
    events = []
    cursor = start_time

    pitches = list(row_form)

    while cursor < start_time + duration:
        for pc in pitches:
            if cursor >= start_time + duration:
                break

            if waltz:
                beat_in_group = len(events) % 3
                if beat_in_group == 0:
                    tone_dur = rng.uniform(2.5, 4.0) * tempo_mult
                else:
                    tone_dur = rng.uniform(1.0, 2.0) * tempo_mult
            else:
                tone_dur = rng.uniform(2.0, 5.0) * tempo_mult

            if rng.random() < 0.15:
                octave = octave_center + rng.choice([-2, -1, 1, 2])
            else:
                octave = octave_center + rng.choice([-1, 0, 0, 0, 1])
            octave = max(2, min(6, octave))

            phrase_pos = (cursor - start_time) / max(duration, 1)

            if arc_shape == "bell":
                arc = np.sin(phrase_pos * np.pi)
            elif arc_shape == "rise":
                arc = phrase_pos ** 0.7
            elif arc_shape == "fall":
                arc = (1.0 - phrase_pos) ** 0.7
            elif arc_shape == "plateau":
                if phrase_pos < 0.15:
                    arc = phrase_pos / 0.15
                elif phrase_pos > 0.85:
                    arc = (1.0 - phrase_pos) / 0.15
                else:
                    arc = 1.0
            else:
                arc = 0.7

            amp = amp_base + (amp_peak - amp_base) * arc
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

            advance = tone_dur * rng.uniform(0.4, 0.8)
            cursor += advance

    return events


def make_tonal_window(rng, triad_pcs, start_time, duration,
                      timbre_strings, timbre_winds, timbre_colors):
    """Create a tonal 'window' — Berg's moment of near-resolution."""
    events = []

    # Root position triad in low strings
    for i, pc in enumerate(triad_pcs):
        t = start_time + i * rng.uniform(1.0, 2.0)
        events.append({
            "time": t,
            "pc": pc,
            "octave": 3,
            "duration": rng.uniform(6.0, 10.0),
            "amplitude": rng.uniform(0.10, 0.16),
            "timbre": rng.choice(timbre_strings)
        })
        # Octave doubling
        events.append({
            "time": t + rng.uniform(0.2, 0.5),
            "pc": pc,
            "octave": 4,
            "duration": rng.uniform(5.0, 8.0),
            "amplitude": rng.uniform(0.06, 0.10),
            "timbre": rng.choice(timbre_winds)
        })

    # Color tones — shimmer
    n_colors = rng.randint(3, 6)
    for _ in range(n_colors):
        events.append({
            "time": rng.uniform(start_time + 2, start_time + duration - 2),
            "pc": rng.choice(triad_pcs),
            "octave": rng.choice([5, 6]),
            "duration": rng.uniform(3.0, 6.0),
            "amplitude": rng.uniform(0.03, 0.07),
            "timbre": rng.choice(timbre_colors)
        })

    return events


def generate_berg_extended(seed=None):
    """Generate the extended Berg piece — 7 cycles of development."""

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)
    all_events = []

    # Row forms at different transpositions for each cycle
    transpositions = [0, 5, 3, 7, 2, 10, 0]  # circle of related keys
    # Triads embedded in each transposition:
    # T0: Gm (7,10,2)  T5: Cm (0,3,7)  T3: Bbm (10,1,5)
    # T7: Dm (2,5,9)   T2: Am (9,0,4)  T10: F#m (6,9,1)  T0: back home

    triads = {
        0:  [7, 10, 2],   # G minor
        5:  [0, 3, 7],    # C minor
        3:  [10, 1, 5],   # Bb minor
        7:  [2, 5, 9],    # D minor
        2:  [9, 0, 4],    # A minor
        10: [6, 9, 1],    # F# minor
    }

    # ================================================================
    # CYCLE 1: EMERGENCE (0-60s)
    # pp, 2 string voices, slow, dark
    # ================================================================
    print("  Cycle 1: Emergence (0-60s)")
    row = row_transpose(BERG_ROW, transpositions[0])

    all_events.extend(make_legato_line(
        rng, row, 1.0, 55, octave_center=3,
        amp_base=0.04, amp_peak=0.10,
        timbre_pool=STRINGS, tempo_mult=1.3, arc_shape="rise"
    ))
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 6.0, 50, octave_center=4,
        amp_base=0.03, amp_peak=0.08,
        timbre_pool=STRINGS, tempo_mult=1.2, arc_shape="rise"
    ))

    # ================================================================
    # CYCLE 2: AWAKENING (55-120s)
    # p, 3 voices, winds enter, slightly more motion
    # ================================================================
    print("  Cycle 2: Awakening (55-120s)")
    row = row_transpose(BERG_ROW, transpositions[1])

    all_events.extend(make_legato_line(
        rng, row, 55.0, 60, octave_center=3,
        amp_base=0.06, amp_peak=0.14,
        timbre_pool=STRINGS, tempo_mult=1.1, arc_shape="bell"
    ))
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 60.0, 55, octave_center=4,
        amp_base=0.05, amp_peak=0.12,
        timbre_pool=WINDS, tempo_mult=1.0, arc_shape="bell"
    ))
    all_events.extend(make_legato_line(
        rng, row_retrograde(row), 65.0, 50, octave_center=5,
        amp_base=0.03, amp_peak=0.08,
        timbre_pool=[TIMBRES[2]],  # solo flute
        tempo_mult=1.1, arc_shape="rise"
    ))

    # First tonal window — brief (C minor)
    all_events.extend(make_tonal_window(
        rng, triads[5], 100.0, 12, STRINGS, WINDS, COLORS
    ))

    # ================================================================
    # CYCLE 3: WALTZ (115-180s)
    # mp, waltz rhythm, full ensemble, Berg's signature
    # ================================================================
    print("  Cycle 3: Waltz (115-180s)")
    row = row_transpose(BERG_ROW, transpositions[2])

    # Main waltz voice — clarinet
    all_events.extend(make_legato_line(
        rng, row, 115.0, 60, octave_center=4,
        amp_base=0.08, amp_peak=0.20,
        timbre_pool=[TIMBRES[3]],  # clarinet
        waltz=True, tempo_mult=0.9, arc_shape="bell"
    ))
    # String accompaniment — lower, waltz bass
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 118.0, 55, octave_center=3,
        amp_base=0.06, amp_peak=0.16,
        timbre_pool=STRINGS, waltz=True, tempo_mult=0.9, arc_shape="plateau"
    ))
    # High wind counterpoint
    all_events.extend(make_legato_line(
        rng, row_retrograde(row), 122.0, 50, octave_center=5,
        amp_base=0.04, amp_peak=0.12,
        timbre_pool=[TIMBRES[2], TIMBRES[7]],  # flute, oboe
        tempo_mult=1.0, arc_shape="bell"
    ))
    # Color accents — bell and glass
    all_events.extend(make_legato_line(
        rng, row, 125.0, 45, octave_center=5,
        amp_base=0.02, amp_peak=0.06,
        timbre_pool=COLORS, tempo_mult=1.5, arc_shape="bell"
    ))

    # ================================================================
    # CYCLE 4: TONAL HEART (175-250s) — the emotional center
    # mf, extended tonal windows, maximum warmth
    # ================================================================
    print("  Cycle 4: Tonal Heart (175-250s)")
    row = row_transpose(BERG_ROW, transpositions[3])

    # Lush strings — wide, sustained
    all_events.extend(make_legato_line(
        rng, row, 175.0, 70, octave_center=3,
        amp_base=0.08, amp_peak=0.22,
        timbre_pool=STRINGS, tempo_mult=1.3, arc_shape="bell"
    ))
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 178.0, 65, octave_center=4,
        amp_base=0.07, amp_peak=0.18,
        timbre_pool=STRINGS + WINDS, tempo_mult=1.2, arc_shape="bell"
    ))

    # Extended D minor tonal window (the heart of the piece)
    all_events.extend(make_tonal_window(
        rng, triads[7], 200.0, 20, STRINGS, WINDS, COLORS
    ))

    # High melody over the tonal window
    all_events.extend(make_legato_line(
        rng, row_retrograde(row), 205.0, 30, octave_center=5,
        amp_base=0.06, amp_peak=0.16,
        timbre_pool=[TIMBRES[7], TIMBRES[2]],  # oboe, flute
        tempo_mult=1.0, arc_shape="fall"
    ))

    # Second tonal window — A minor (deepening)
    all_events.extend(make_tonal_window(
        rng, triads[2], 230.0, 15, STRINGS, WINDS, COLORS
    ))

    # Pizzicato texture underneath (something new — ear candy)
    for i in range(10):
        t_ev = 195.0 + i * rng.uniform(3.0, 6.0)
        if t_ev > 248:
            break
        pc = int(row[i % 12])
        all_events.append({
            "time": t_ev,
            "pc": pc,
            "octave": rng.choice([2, 3]),
            "duration": rng.uniform(0.5, 1.5),
            "amplitude": rng.uniform(0.04, 0.08),
            "timbre": TIMBRES[6]  # pizzicato
        })

    # ================================================================
    # CYCLE 5: STORM (245-310s) — climax
    # f, 5 dense voices, dramatic, maximum intensity
    # ================================================================
    print("  Cycle 5: Storm (245-310s)")
    row = row_transpose(BERG_ROW, transpositions[4])

    # Five simultaneous voices — maximum density
    all_events.extend(make_legato_line(
        rng, row, 245.0, 60, octave_center=2,
        amp_base=0.10, amp_peak=0.28,
        timbre_pool=[TIMBRES[0]],  # cello pont (intense)
        tempo_mult=0.8, arc_shape="plateau"
    ))
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 247.0, 58, octave_center=3,
        amp_base=0.08, amp_peak=0.24,
        timbre_pool=STRINGS, tempo_mult=0.7, arc_shape="plateau"
    ))
    all_events.extend(make_legato_line(
        rng, row_retrograde(row), 248.0, 56, octave_center=4,
        amp_base=0.08, amp_peak=0.22,
        timbre_pool=WINDS, waltz=True, tempo_mult=0.7, arc_shape="plateau"
    ))
    all_events.extend(make_legato_line(
        rng, row_transpose(row_inversion(row), 6), 250.0, 54, octave_center=5,
        amp_base=0.06, amp_peak=0.18,
        timbre_pool=WINDS, tempo_mult=0.8, arc_shape="bell"
    ))
    # Bells punctuating the storm
    all_events.extend(make_legato_line(
        rng, row, 252.0, 48, octave_center=5,
        amp_base=0.04, amp_peak=0.10,
        timbre_pool=COLORS, tempo_mult=1.0, arc_shape="bell"
    ))

    # Dramatic tonal window at climax — F# minor (remote, anguished)
    all_events.extend(make_tonal_window(
        rng, triads[10], 280.0, 15, STRINGS, WINDS, COLORS
    ))

    # ================================================================
    # CYCLE 6: REMEMBRANCE (305-370s)
    # mp→pp, echoes of earlier material, nostalgic
    # ================================================================
    print("  Cycle 6: Remembrance (305-370s)")
    row = row_transpose(BERG_ROW, transpositions[5])

    # Two voices — echo of Cycle 1, but wiser, sadder
    all_events.extend(make_legato_line(
        rng, row, 305.0, 60, octave_center=3,
        amp_base=0.06, amp_peak=0.14,
        timbre_pool=[TIMBRES[1]],  # cello tasto only (dark)
        tempo_mult=1.3, arc_shape="fall"
    ))
    all_events.extend(make_legato_line(
        rng, row_inversion(row), 310.0, 55, octave_center=4,
        amp_base=0.04, amp_peak=0.10,
        timbre_pool=[TIMBRES[2]],  # solo flute (distant)
        tempo_mult=1.4, arc_shape="fall"
    ))

    # Echo of the waltz — fragmentary, slower, quieter
    all_events.extend(make_legato_line(
        rng, row_retrograde(row), 330.0, 30, octave_center=4,
        amp_base=0.03, amp_peak=0.08,
        timbre_pool=[TIMBRES[3]],  # clarinet alone
        waltz=True, tempo_mult=1.5, arc_shape="fall"
    ))

    # One last tonal window — back to Gm, the home triad
    all_events.extend(make_tonal_window(
        rng, triads[0], 345.0, 15, STRINGS, WINDS, COLORS
    ))

    # ================================================================
    # CYCLE 7: DISSOLUTION (365-420s)
    # ppp, single voice, evaporating into silence
    # ================================================================
    print("  Cycle 7: Dissolution (365-420s)")
    row = row_transpose(BERG_ROW, transpositions[6])  # back to T0

    # Single voice — glass harmonica (ethereal)
    all_events.extend(make_legato_line(
        rng, row, 365.0, 45, octave_center=4,
        amp_base=0.02, amp_peak=0.06,
        timbre_pool=[TIMBRES[5]],  # glass harmonica only
        tempo_mult=1.8, arc_shape="fall"
    ))

    # Isolated final tones — the piece evaporating
    final_pcs = [7, 10, 2]  # G, Bb, D — G minor, where it all began
    for i, pc in enumerate(final_pcs):
        all_events.append({
            "time": 395.0 + i * 4.0,
            "pc": pc,
            "octave": rng.choice([5, 6]),
            "duration": rng.uniform(4.0, 8.0),
            "amplitude": 0.025 - i * 0.005,
            "timbre": rng.choice(COLORS)
        })

    # Very last note — G3, cello tasto, pppp
    all_events.append({
        "time": 410.0,
        "pc": 7,  # G
        "octave": 3,
        "duration": 10.0,
        "amplitude": 0.02,
        "timbre": TIMBRES[1]
    })

    # ================================================================
    # RENDER
    # ================================================================
    print(f"\n  Rendering {len(all_events)} events across {DURATION}s...")

    for event in all_events:
        audio += pointillist_tone(
            t, event["time"], event["pc"], event["octave"],
            event["duration"], event["amplitude"], event["timbre"],
            np_rng
        )

    # Reverb — concert hall, generous
    print("  Applying reverb...")
    audio = simple_reverb(audio, decay=0.50, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 2.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 5.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.1)

    # Gentle saturation
    audio = np.tanh(audio * 1.05) / 1.05

    return audio, all_events


def main():
    parser = argparse.ArgumentParser(
        description="Extended Berg piece — 7 minutes of lyrical serialism"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="berg_extended_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating extended Berg piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s ({DURATION // 60}m {DURATION % 60}s)")
    print(f"  Structure: 7 cycles through transposed material\n")

    audio, events = generate_berg_extended(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")

    # Cycle summary
    cycle_bounds = [
        ("Emergence", 0, 60),
        ("Awakening", 55, 120),
        ("Waltz", 115, 180),
        ("Tonal Heart", 175, 250),
        ("Storm", 245, 310),
        ("Remembrance", 305, 370),
        ("Dissolution", 365, 420),
    ]
    for name, s, e in cycle_bounds:
        n = sum(1 for ev in events if s <= ev["time"] < e)
        print(f"  {name:15s} ({s:3d}-{e:3d}s): {n:3d} events")


if __name__ == "__main__":
    main()
