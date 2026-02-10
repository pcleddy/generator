"""
tubular_low.py — Pure tubular bells, low register, 1 minute

Just the good stuff. Nothing else. Deep tolling with that
tierce partial doing the heavy lifting.

Usage:
    python tubular_low.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import random
import argparse

from webern_pointillism import SAMPLE_RATE, simple_reverb
from bells_bergman import bell_strike, freq_from_pc, generate_bell_layer

DURATION = 70  # a little breathing room on each end

# D minor pentatonic: D F G A C — dark, resonant, no tension
D_MIN_PENT = [2, 5, 7, 9, 0]

# D natural minor for more melodic passages: D E F G A Bb C
D_MINOR = [2, 4, 5, 7, 9, 10, 0]


def generate_tubular_low(seed=None):
    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)
    events = []

    # ===========================================
    # LAYER 1: SLOW TOLLING (whole piece)
    # Deep, sparse, single bells — the backbone
    # ===========================================
    print("  Layer 1: Slow tolling...")
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MIN_PENT, octave_range=[2, 3],
        profile_name="tubular_bell",
        start_time=3, end_time=65,
        density=0.10,
        amp_range=(0.06, 0.09),
        dur_range=(6.0, 12.0),
        pattern="melodic"
    )

    # ===========================================
    # LAYER 2: OVERLAPPING RINGS (15-55s)
    # Slightly higher, building texture from overlap
    # ===========================================
    print("  Layer 2: Overlapping rings...")
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=15, end_time=55,
        density=0.15,
        amp_range=(0.04, 0.07),
        dur_range=(5.0, 10.0),
        pattern="melodic"
    )

    # ===========================================
    # LAYER 3: ARPEGGIATED TOLLING (25-50s)
    # Triadic arpeggios — broken chords ringing
    # ===========================================
    print("  Layer 3: Arpeggiated tolling...")
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[2, 3, 4],
        profile_name="tubular_bell",
        start_time=25, end_time=50,
        density=0.12,
        amp_range=(0.05, 0.08),
        dur_range=(6.0, 10.0),
        pattern="arpeggic"
    )

    # ===========================================
    # LAYER 4: CHORDAL SWELLS (35-48s)
    # Peak density — multiple bells ringing together
    # ===========================================
    print("  Layer 4: Chordal swells...")
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MIN_PENT, octave_range=[2, 3],
        profile_name="tubular_bell",
        start_time=35, end_time=48,
        density=0.10,
        amp_range=(0.05, 0.08),
        dur_range=(7.0, 12.0),
        pattern="chordal"
    )

    # ===========================================
    # ANCHOR BELLS — specific low tolls
    # ===========================================
    print("  Anchor bells...")

    # Opening toll — D2
    audio += bell_strike(t, 2.0, freq_from_pc(2, 2), 12.0, 0.07,
                        "tubular_bell", np_rng)
    events.append({'time': 2.0, 'pc': 2, 'octave': 2, 'dur': 12.0,
                  'amp': 0.07, 'profile': 'tubular_bell', 'pattern': 'anchor'})

    # Midpoint — A2 (the fifth, open)
    audio += bell_strike(t, 33.0, freq_from_pc(9, 2), 14.0, 0.08,
                        "tubular_bell", np_rng)
    events.append({'time': 33.0, 'pc': 9, 'octave': 2, 'dur': 14.0,
                  'amp': 0.08, 'profile': 'tubular_bell', 'pattern': 'anchor'})

    # Closing — D2 again, coming home
    audio += bell_strike(t, 58.0, freq_from_pc(2, 2), 14.0, 0.07,
                        "tubular_bell", np_rng)
    events.append({'time': 58.0, 'pc': 2, 'octave': 2, 'dur': 14.0,
                  'amp': 0.07, 'profile': 'tubular_bell', 'pattern': 'anchor'})

    # ===========================================
    # MIX
    # ===========================================
    print("  Mixing...")

    # Cathedral reverb — long, deep
    cathedral_delays = [37, 53, 71, 97, 131, 173, 229, 307, 401, 503]
    audio = simple_reverb(audio, decay=0.60, delays_ms=cathedral_delays,
                         sample_rate=SAMPLE_RATE)

    # Gentle envelope
    fade_in = np.minimum(t / 3.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 5.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize — keep it warm, not loud
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.15)

    audio = np.tanh(audio * 1.02) / 1.02

    return audio, events


def main():
    parser = argparse.ArgumentParser(
        description="Pure tubular bells — low register, 1 minute"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="tubular_low_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating tubular bells piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Timbre: 100% tubular bell")
    print(f"  Register: octaves 2-4\n")

    audio, events = generate_tubular_low(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")


if __name__ == "__main__":
    main()
