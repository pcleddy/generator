"""
bells_pizz.py — Bells variation: pizzicato tick-tock + heavy tubular bells

Variation on bells_bergman with:
  - Pizzicato patterns as the "clock" (deeper, more musical tick-tock)
  - More prominent tubular bells
  - Shorter sampler (~90 seconds)

Usage:
    python bells_pizz.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import random
import argparse

from webern_pointillism import (
    SAMPLE_RATE, TIMBRES, pointillist_tone, simple_reverb, freq_from_pitch_class
)
from bells_bergman import (
    bell_strike, BELL_PROFILES, freq_from_pc,
    generate_bell_layer, D_DORIAN, D_MINOR, G_MAJOR
)

DURATION = 90
BASE_FREQ = 261.63

# E Phrygian: E F G A B C D — dark, Spanish, pairs well with bells
E_PHRYGIAN = [4, 5, 7, 9, 11, 0, 2]

# A minor pentatonic for the pizz patterns: A C D E G
A_PENT = [9, 0, 2, 4, 7]


def make_pizz_clock(t, rng, np_rng, bpm=72, amplitude=0.06):
    """Pizzicato tick-tock — deeper, more musical than mechanical clock.

    Alternating patterns:
      - Low pizz "tick" (cello range)
      - Slightly higher pizz "tock"
      - Every 4th beat: a little two-note figure
      - Every 8th beat: a three-note turn
    """
    audio = np.zeros_like(t)
    period = 60.0 / bpm
    pizz = TIMBRES[6]  # pizzicato timbre

    # Two alternating pitches from A minor pentatonic
    tick_pc, tick_oct = 9, 2    # A2 — deep
    tock_pc, tock_oct = 4, 3    # E3 — a fifth above

    n_beats = int(DURATION / period) + 1
    events = []

    for i in range(n_beats):
        beat_time = i * period
        if beat_time >= DURATION - 0.5:
            break

        beat_in_bar = i % 8

        if beat_in_bar == 0:
            # Strong beat — deeper, louder
            audio += pointillist_tone(
                t, beat_time, tick_pc, tick_oct,
                0.8, amplitude * 1.1, pizz, np_rng
            )
            events.append({'time': beat_time, 'pc': tick_pc, 'oct': tick_oct, 'type': 'tick'})

        elif beat_in_bar == 4:
            # Secondary strong beat
            audio += pointillist_tone(
                t, beat_time, tock_pc, tock_oct - 1,
                0.6, amplitude * 0.9, pizz, np_rng
            )
            events.append({'time': beat_time, 'pc': tock_pc, 'oct': tock_oct - 1, 'type': 'tock'})

        elif beat_in_bar == 3 or beat_in_bar == 7:
            # Two-note pickup figure
            pickup_pc = rng.choice(A_PENT)
            audio += pointillist_tone(
                t, beat_time, pickup_pc, 3,
                0.4, amplitude * 0.5, pizz, np_rng
            )
            audio += pointillist_tone(
                t, beat_time + period * 0.5, rng.choice(A_PENT), 3,
                0.3, amplitude * 0.4, pizz, np_rng
            )
            events.append({'time': beat_time, 'pc': pickup_pc, 'oct': 3, 'type': 'figure'})

        elif beat_in_bar == 6:
            # Three-note turn (ornamental)
            turn_root = rng.choice(A_PENT)
            for j, (pc_offset, delay) in enumerate([(0, 0), (2, 0.12), (0, 0.24)]):
                pc = A_PENT[(A_PENT.index(turn_root) + pc_offset) % len(A_PENT)]
                audio += pointillist_tone(
                    t, beat_time + delay, pc, 3,
                    0.35, amplitude * 0.45 * (0.9 ** j), pizz, np_rng
                )
            events.append({'time': beat_time, 'pc': turn_root, 'oct': 3, 'type': 'turn'})

        else:
            # Regular alternating tick-tock
            is_tick = (i % 2 == 0)
            pc = tick_pc if is_tick else tock_pc
            oct = tick_oct if is_tick else tock_oct
            amp = amplitude * (0.7 if is_tick else 0.55)

            audio += pointillist_tone(
                t, beat_time, pc, oct,
                0.5, amp, pizz, np_rng
            )
            events.append({'time': beat_time, 'pc': pc, 'oct': oct,
                          'type': 'tick' if is_tick else 'tock'})

    return audio, events


def generate_bells_pizz(seed=None):
    """Generate the bells + pizzicato variation."""

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)
    all_events = []

    # ===========================================
    # PIZZICATO CLOCK (always present)
    # ===========================================
    print("  Building pizzicato clock...")
    pizz_audio, pizz_events = make_pizz_clock(t, rng, np_rng, bpm=72, amplitude=0.055)

    # Fade in the clock
    pizz_fade = np.minimum(t / 3.0, 1.0)
    pizz_audio *= pizz_fade

    # ===========================================
    # SECTION 1: PIZZ ALONE (0-12s)
    # ===========================================
    print("  Section 1: Pizz alone (0-12s)")
    # Just the pizzicato establishing its pattern
    # One distant tubular bell at 8s
    audio += bell_strike(t, 8.0, freq_from_pc(9, 3), 6.0, 0.04,
                        "tubular_bell", np_rng)
    all_events.append({'time': 8.0, 'pc': 9, 'octave': 3, 'dur': 6.0,
                      'amp': 0.04, 'profile': 'tubular_bell', 'pattern': 'solo'})

    # ===========================================
    # SECTION 2: TUBULAR BELLS ENTER (12-40s)
    # ===========================================
    print("  Section 2: Tubular bells (12-40s)")

    # Prominent tubular bell melody — E Phrygian
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=E_PHRYGIAN, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=12, end_time=40,
        density=0.22,
        amp_range=(0.06, 0.11),
        dur_range=(4.0, 8.0),
        pattern="melodic"
    )

    # Glockenspiel answering from above
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=E_PHRYGIAN, octave_range=[5, 6],
        profile_name="glockenspiel",
        start_time=18, end_time=38,
        density=0.3,
        amp_range=(0.03, 0.06),
        dur_range=(1.5, 3.0),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 3: LAYERED BELLS (40-65s)
    # ===========================================
    print("  Section 3: Full bells (40-65s)")

    # Heavy tubular bell arpeggios
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=D_MINOR, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=40, end_time=65,
        density=0.28,
        amp_range=(0.07, 0.12),
        dur_range=(4.0, 8.0),
        pattern="arpeggic"
    )

    # Church bell — one deep toll
    audio += bell_strike(t, 48.0, freq_from_pc(2, 2), 12.0, 0.08,
                        "church_bell", np_rng)
    all_events.append({'time': 48.0, 'pc': 2, 'octave': 2, 'dur': 12.0,
                      'amp': 0.08, 'profile': 'church_bell', 'pattern': 'toll'})

    # Celesta arpeggios
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=D_MINOR, octave_range=[4, 5],
        profile_name="celesta",
        start_time=42, end_time=62,
        density=0.4,
        amp_range=(0.03, 0.06),
        dur_range=(2.0, 4.0),
        pattern="arpeggic"
    )

    # Glockenspiel shimmer
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=D_MINOR, octave_range=[5, 6, 7],
        profile_name="glockenspiel",
        start_time=45, end_time=63,
        density=0.5,
        amp_range=(0.025, 0.05),
        dur_range=(1.0, 2.5),
        pattern="melodic"
    )

    # Tubular bell chords (the money shot)
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=D_MINOR, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=52, end_time=63,
        density=0.15,
        amp_range=(0.07, 0.11),
        dur_range=(5.0, 9.0),
        pattern="chordal"
    )

    # ===========================================
    # SECTION 4: THINNING (65-90s)
    # ===========================================
    print("  Section 4: Thinning (65-90s)")

    # Tubular bells continue but sparser
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=E_PHRYGIAN, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=65, end_time=82,
        density=0.12,
        amp_range=(0.05, 0.08),
        dur_range=(5.0, 9.0),
        pattern="melodic"
    )

    # Wind chimes — delicate exit
    audio += generate_bell_layer(
        t, all_events, rng, np_rng,
        scale=E_PHRYGIAN, octave_range=[6, 7],
        profile_name="wind_chime",
        start_time=70, end_time=85,
        density=0.3,
        amp_range=(0.01, 0.025),
        dur_range=(0.5, 1.5),
        pattern="melodic"
    )

    # Final tubular bell toll — E2
    audio += bell_strike(t, 80.0, freq_from_pc(4, 2), 10.0, 0.06,
                        "tubular_bell", np_rng)
    all_events.append({'time': 80.0, 'pc': 4, 'octave': 2, 'dur': 10.0,
                      'amp': 0.06, 'profile': 'tubular_bell', 'pattern': 'final'})

    # ===========================================
    # MIX
    # ===========================================
    print("  Mixing...")
    audio += pizz_audio

    # Cathedral reverb
    cathedral_delays = [31, 47, 67, 89, 113, 149, 191, 251, 313, 397]
    audio = simple_reverb(audio, decay=0.55, delays_ms=cathedral_delays,
                         sample_rate=SAMPLE_RATE)

    # Envelope
    fade_in = np.minimum(t / 2.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 4.0, 1.0)
    fade_out = np.maximum(fade_out, 0.1)  # pizz persists
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.12)

    audio = np.tanh(audio * 1.02) / 1.02

    return audio, all_events


def main():
    parser = argparse.ArgumentParser(
        description="Bells variation with pizzicato clock + heavy tubular bells"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="bells_pizz_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating bells + pizzicato variation...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s\n")

    audio, events = generate_bells_pizz(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")
    print(f"  Total bell events: {len(events)}")


if __name__ == "__main__":
    main()
