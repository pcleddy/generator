"""
cage_ambient.py — Ambient drift with John Cage-style chance interruptions

A long, slow ambient bed (borrowing from ambient_synthesis.py's approach)
periodically SLAPPED by violent prepared-piano clusters, tone smashes,
and percussive chaos. Cage's chance operations determine when and what.

Structure:
  - Base layer: continuous 12-tone ambient drift (quiet, evolving)
  - Interruptions: 5-8 sudden violent events at random times
    - Prepared piano: metallic, detuned inharmonic resonances
    - Tone clusters: forearm-on-keyboard, 8-15 simultaneous notes
    - Piano slaps: percussive broadband + resonant strings
    - Silence after each interruption (the shock of absence)
  - Duration: 150 seconds — needs room for the tension/release cycle

The ambient sections should make you forget anything can happen.
Then it happens.

Usage:
    python cage_ambient.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import random
import argparse

from webern_pointillism import (
    SAMPLE_RATE, TIMBRES as INST_TIMBRES, pointillist_tone,
    simple_reverb, freq_from_pitch_class
)
from berg_vocal import vocal_tone, VOICE_TIMBRES

DURATION = 150
BASE_FREQ = 261.63


# =====================================================================
# PREPARED PIANO SYNTHESIS
# =====================================================================

def prepared_piano_strike(t, start, pitch_class, octave, duration,
                          amplitude, preparation, rng):
    """Synthesize a prepared piano string being struck.

    Prepared piano: objects placed on strings (bolts, screws, rubber,
    felt) create inharmonic, metallic, buzzing, or muted timbres.
    Each "preparation" type changes the harmonic behavior.

    Signal chain:
      1. Percussive attack (hammer impact noise)
      2. Inharmonic partials (detuned by preparation objects)
      3. Buzzing/rattling modulation
      4. Fast decay (preparations damp the string)
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = BASE_FREQ * (2 ** ((pitch_class + octave * 12 - 48) / 12))

    if freq < 25 or freq > 6000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)

    signal = np.zeros(n_active)

    if preparation == "bolt":
        # Bolt on strings: metallic, many inharmonic partials, buzzy
        n_partials = rng.randint(8, 14)
        for i in range(n_partials):
            # Partials are NOT integer multiples — bolt creates chaos
            partial_ratio = (i + 1) * (1.0 + rng.uniform(-0.08, 0.08))
            partial_freq = freq * partial_ratio
            if partial_freq > SAMPLE_RATE / 2 - 100:
                continue

            phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)
            partial_amp = 1.0 / (i + 1) ** 0.8  # slow rolloff = bright/metallic

            # Each partial decays at different rate
            decay = np.exp(-(2.0 + i * 0.6 + rng.uniform(0, 1.0)) * t_norm)

            # Buzz: amplitude modulation from bolt rattling
            buzz_freq = rng.uniform(30, 120)
            buzz = 1.0 + 0.3 * np.sin(2 * np.pi * buzz_freq * t_local)

            signal += partial_amp * np.sin(phase) * decay * buzz

    elif preparation == "rubber":
        # Rubber mute: dark, thuddy, very fast decay, few partials
        n_partials = rng.randint(3, 6)
        for i in range(n_partials):
            partial_ratio = (i + 1) * (1.0 + rng.uniform(-0.02, 0.02))
            partial_freq = freq * partial_ratio
            if partial_freq > SAMPLE_RATE / 2 - 100:
                continue

            phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)
            partial_amp = 1.0 / (i + 1) ** 1.5  # fast rolloff = dark

            # Very fast decay — rubber absorbs energy
            decay = np.exp(-(6.0 + i * 2.0) * t_norm)
            signal += partial_amp * np.sin(phase) * decay

    elif preparation == "screw":
        # Screw between strings: jangling, bright, sustained rattle
        n_partials = rng.randint(10, 18)
        for i in range(n_partials):
            # Wild detuning — screw creates multiple contact points
            partial_ratio = (i + 1) * (1.0 + rng.uniform(-0.15, 0.15))
            partial_freq = freq * partial_ratio
            if partial_freq > SAMPLE_RATE / 2 - 100:
                continue

            phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)
            partial_amp = rng.uniform(0.3, 1.0) / (i + 1) ** 0.5

            decay = np.exp(-(1.5 + i * 0.3) * t_norm)

            # Screw rattle: irregular amplitude modulation
            rattle_freq = rng.uniform(15, 80)
            rattle = 1.0 + 0.5 * np.sin(2 * np.pi * rattle_freq * t_local +
                                          0.3 * np.sin(2 * np.pi * rattle_freq * 0.7 * t_local))
            signal += partial_amp * np.sin(phase) * decay * rattle

    # Normalize partials
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # --- HAMMER IMPACT: percussive noise burst ---
    impact_len = min(int(rng.uniform(0.003, 0.015) * SAMPLE_RATE), n_active)
    impact_noise = rng.randn(n_active)
    impact_env = np.zeros(n_active)
    if impact_len > 0:
        impact_env[:impact_len] = np.exp(-np.linspace(0, 12, impact_len))
    impact = impact_noise * impact_env * 0.4

    # Combine
    combined = signal * 0.7 + impact * 0.3

    # Overall envelope
    attack_len = max(int(0.001 * SAMPLE_RATE), 1)
    envelope = np.ones(n_active)
    envelope[:min(attack_len, n_active)] = np.linspace(0, 1, min(attack_len, n_active))
    # Anti-click fadeout
    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        envelope[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = combined * envelope * amplitude
    return voice


def tone_cluster(t, start, center_pc, octave, n_notes, spread_semitones,
                 duration, amplitude, preparation, rng):
    """Generate a tone cluster: many notes smashed simultaneously.

    Cowell/Cage: forearm on keyboard, palm slap, elbow smash.
    Notes are densely packed within a chromatic range.
    """
    cluster = np.zeros_like(t)

    for i in range(n_notes):
        # Spread notes around center pitch
        offset = rng.uniform(-spread_semitones / 2, spread_semitones / 2)
        note_pc = (center_pc + offset) % 12
        note_oct = octave + int(offset // 12)
        note_oct = max(1, min(7, note_oct))

        # Slight timing spread — not perfectly simultaneous (like a real slam)
        time_offset = rng.uniform(0, 0.03)

        # Individual note duration varies slightly
        note_dur = duration * rng.uniform(0.6, 1.0)

        # Amplitude varies per note
        note_amp = amplitude * rng.uniform(0.5, 1.0)

        cluster += prepared_piano_strike(
            t, start + time_offset, int(note_pc), note_oct,
            note_dur, note_amp, preparation, rng
        )

    return cluster


# =====================================================================
# AMBIENT BED — slow, evolving 12-tone drift
# =====================================================================

def make_ambient_bed(t, rng, np_rng):
    """Generate continuous ambient bed using the instrument engine."""
    audio = np.zeros_like(t)

    twelve_tone = np.array([0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10])
    inv = (12 - twelve_tone) % 12

    # Slow primary row voices
    for cycle in range(3):  # 3 passes through the row
        for i, pitch in enumerate(twelve_tone):
            start = cycle * 50 + i * 4.2
            if start + 8 > DURATION:
                break
            timbre = random.choice([INST_TIMBRES[0], INST_TIMBRES[1],
                                    INST_TIMBRES[5]])  # strings + glass
            audio += pointillist_tone(
                t, start, int(pitch), octave=3,
                duration=rng.uniform(6, 12), amplitude=rng.uniform(0.04, 0.07),
                timbre=timbre, rng=np_rng
            )

    # Inverted row, higher register, quieter
    for cycle in range(2):
        for i, pitch in enumerate(inv):
            start = 5 + cycle * 55 + i * 4.8
            if start + 8 > DURATION:
                break
            timbre = random.choice([INST_TIMBRES[2], INST_TIMBRES[7]])  # flute, oboe
            audio += pointillist_tone(
                t, start, int(pitch), octave=4,
                duration=rng.uniform(5, 10), amplitude=rng.uniform(0.02, 0.045),
                timbre=timbre, rng=np_rng
            )

    # Sub-bass drone
    for start in [0, 40, 80, 120]:
        audio += pointillist_tone(
            t, start, 0, octave=2,
            duration=35, amplitude=0.03,
            timbre=INST_TIMBRES[1], rng=np_rng  # cello tasto
        )

    return audio


# =====================================================================
# CAGE INTERRUPTIONS — chance-determined violent events
# =====================================================================

def generate_interruptions(t, rng, np_rng):
    """Generate Cage-style chance interruptions.

    Each interruption is:
      1. A sudden violent sonic event (cluster, slap, prepared piano)
      2. Followed by enforced silence (duck the ambient bed)

    Timing determined by chance operations (random).
    """
    audio = np.zeros_like(t)
    duck_envelope = np.ones_like(t)  # 1.0 = full ambient, 0.0 = silenced

    # Number of interruptions: 5-8 across the piece
    n_interruptions = rng.randint(5, 8)

    # Minimum spacing: 12 seconds (need ambient to re-establish)
    # Don't interrupt in first 15s or last 10s
    available_window = DURATION - 25
    interruption_times = sorted(rng.sample(
        [x * 0.5 for x in range(30, int(available_window * 2))],
        min(n_interruptions, int(available_window / 12))
    ))

    # Enforce minimum spacing
    filtered_times = [interruption_times[0]]
    for it in interruption_times[1:]:
        if it - filtered_times[-1] >= 12:
            filtered_times.append(it)
    interruption_times = filtered_times

    print(f"\n  Interruptions ({len(interruption_times)}):")

    for i, int_time in enumerate(interruption_times):
        # Choose interruption type
        int_type = rng.choice([
            "cluster_bolt", "cluster_screw", "single_slap",
            "cascade", "cluster_rubber", "vocal_scream"
        ])

        print(f"    {int_time:6.1f}s  {int_type}")

        if int_type.startswith("cluster"):
            prep = int_type.split("_")[1]
            center = rng.randint(0, 11)
            n_notes = rng.randint(8, 15)
            spread = rng.uniform(8, 18)  # semitones
            dur = rng.uniform(1.5, 4.0)
            amp = rng.uniform(0.20, 0.35)

            audio += tone_cluster(
                t, int_time, center, rng.choice([2, 3, 4]),
                n_notes, spread, dur, amp, prep, np_rng
            )

        elif int_type == "single_slap":
            # One massive prepared piano strike
            audio += prepared_piano_strike(
                t, int_time, rng.randint(0, 11), rng.choice([2, 3]),
                rng.uniform(2, 5), rng.uniform(0.25, 0.40),
                rng.choice(["bolt", "screw"]), np_rng
            )

        elif int_type == "cascade":
            # Rapid succession of strikes — like running forearm down keyboard
            n_strikes = rng.randint(6, 12)
            for j in range(n_strikes):
                strike_time = int_time + j * rng.uniform(0.05, 0.15)
                pc = (j * rng.choice([1, 2, 3])) % 12  # chromatic or whole-tone cascade
                audio += prepared_piano_strike(
                    t, strike_time, pc, rng.choice([2, 3, 4, 5]),
                    rng.uniform(0.8, 2.5), rng.uniform(0.10, 0.25),
                    rng.choice(["bolt", "screw", "rubber"]), np_rng
                )

        elif int_type == "vocal_scream":
            # Brief, intense vocal burst — the human element
            voice_t = rng.choice(VOICE_TIMBRES)
            # Override vowel to "ah" — open scream
            scream_timbre = voice_t.copy()
            scream_timbre["vowel_sequence"] = ["ah", "eh", "ah"]
            scream_timbre["vibrato_depth"] = 25  # wide, frantic
            scream_timbre["vibrato_rate"] = 6.5
            scream_timbre["breathiness"] = 0.12

            oct = scream_timbre["octave_range"][1]  # top of range
            audio += vocal_tone(
                t, int_time, rng.randint(0, 11), oct,
                rng.uniform(1.5, 3.0), rng.uniform(0.18, 0.30),
                scream_timbre, np_rng
            )

        # Duck the ambient bed around each interruption
        # Sharp cut before, gradual recovery after
        duck_start = max(0, int((int_time - 0.1) * SAMPLE_RATE))
        duck_end = min(len(t), int((int_time + 0.3) * SAMPLE_RATE))
        silence_end = min(len(t), int((int_time + rng.uniform(2, 5)) * SAMPLE_RATE))
        recovery_end = min(len(t), silence_end + int(3.0 * SAMPLE_RATE))

        # Rapid duck
        if duck_end > duck_start:
            duck_envelope[duck_start:duck_end] = np.linspace(1, 0.05, duck_end - duck_start)
        # Hold silence
        if silence_end > duck_end:
            duck_envelope[duck_end:silence_end] = 0.05
        # Gradual recovery
        if recovery_end > silence_end:
            duck_envelope[silence_end:recovery_end] = np.linspace(
                0.05, 1.0, recovery_end - silence_end
            )

    return audio, duck_envelope


def generate_cage_piece(seed=None):
    """Generate the ambient + Cage interruption piece."""

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    print("  Building ambient bed...")
    ambient = make_ambient_bed(t, rng, np_rng)

    print("  Generating Cage interruptions...")
    interruptions, duck_envelope = generate_interruptions(t, rng, np_rng)

    # Apply ducking to ambient bed
    ambient *= duck_envelope

    # Combine
    audio = ambient + interruptions

    # Reverb — different amounts for ambient vs interruptions
    # Run the whole thing through a medium reverb
    audio = simple_reverb(audio, decay=0.45, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 3.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 4.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize — preserve dynamic range (interruptions SHOULD be louder)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.05)

    # Light saturation — don't crush the transients
    audio = np.tanh(audio * 1.03) / 1.03

    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Ambient drift with Cage-style chance interruptions"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="cage_ambient_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating Cage/ambient piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")

    audio = generate_cage_piece(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")


if __name__ == "__main__":
    main()
