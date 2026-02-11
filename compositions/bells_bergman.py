"""
bells_bergman.py — Layered glockenspiel/bells with Bergman tick-tock metronome

"Wild Strawberries opens with a dream of a clock without hands."

Structure:
  1. The Clock Alone        (0-15s)   — tick-tock establishes itself in silence
  2. First Bells             (15-40s)  — single glockenspiel enters, D Dorian melody
  3. Accumulation            (40-75s)  — layers build: tubular bell, chimes, celesta
  4. Full Peal               (75-100s) — all bell voices ringing, rich tonal harmony
  5. Dissolution             (100-120s)— bells thin out, clock persists, alone again

Tonality: D Dorian → A Mixolydian → G → D minor (circle of close relations)
The clock never stops. Everything else is mortal.

Bell synthesis: true bell partials (hum, prime, tierce, quint, nominal)
with mode-specific decay rates and strike transients.

Usage:
    python bells_bergman.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import random
import argparse

from webern_pointillism import SAMPLE_RATE, simple_reverb, generate_noise

DURATION = 120
BASE_FREQ = 261.63  # C4


# =====================================================================
# BELL SYNTHESIS — proper bell partials
# =====================================================================

# Real bells have specific partial relationships (not harmonic!):
#   Hum:      ~0.5× fundamental (an octave below)
#   Prime:    1.0× fundamental
#   Tierce:   ~1.183× (minor third above — this is why bells sound "minor")
#   Quint:    ~1.506× (perfect fifth above)
#   Nominal:  ~2.0× (octave above)
#   Deciem:   ~2.514×
#   Undeciem: ~2.662×
#   Duodeciem: ~3.011×

BELL_PROFILES = {
    "glockenspiel": {
        # Small metal bars — bright, high, quick decay
        "partials": [
            (1.0,   1.0,   3.5),   # prime (ratio, amplitude, decay_rate)
            (2.76,  0.45,  4.0),   # ~overtone 1 (metallic, not octave)
            (5.40,  0.25,  5.5),   # overtone 2
            (8.93,  0.12,  7.0),   # overtone 3
            (13.3,  0.06,  9.0),   # overtone 4 — high shimmer
        ],
        "strike_brightness": 8000,
        "strike_amount": 0.20,
        "ring_time_mult": 0.7,  # shorter ring than large bells
    },
    "celesta": {
        # Metal plates over wooden resonators — warm, crystalline
        "partials": [
            (1.0,   1.0,   2.8),
            (2.0,   0.35,  3.5),   # nearly harmonic octave (the resonator helps)
            (3.0,   0.15,  4.5),
            (5.2,   0.08,  6.0),
        ],
        "strike_brightness": 5000,
        "strike_amount": 0.12,
        "ring_time_mult": 1.0,
    },
    "tubular_bell": {
        # Large metal tubes — deep, churchy, long ring
        "partials": [
            (0.5,   0.25,  1.5),   # hum tone — octave below
            (1.0,   1.0,   1.8),   # prime
            (1.183, 0.70,  2.0),   # tierce (minor 3rd — THE bell sound)
            (1.506, 0.45,  2.3),   # quint
            (2.0,   0.55,  2.5),   # nominal (octave)
            (2.514, 0.20,  3.0),   # deciem
            (3.011, 0.12,  3.5),   # duodeciem
            (4.166, 0.05,  5.0),   # upper partial
        ],
        "strike_brightness": 3000,
        "strike_amount": 0.15,
        "ring_time_mult": 2.0,  # long ring
    },
    "church_bell": {
        # Heavy cast bell — deep, massive, very long ring
        "partials": [
            (0.5,   0.35,  1.0),   # strong hum tone
            (1.0,   1.0,   1.2),   # prime
            (1.183, 0.80,  1.3),   # tierce — very prominent
            (1.506, 0.55,  1.5),   # quint
            (2.0,   0.65,  1.6),   # nominal
            (2.514, 0.30,  2.0),
            (2.662, 0.22,  2.2),
            (3.011, 0.15,  2.5),
            (4.166, 0.08,  3.5),
            (5.433, 0.04,  4.5),
        ],
        "strike_brightness": 2000,
        "strike_amount": 0.18,
        "ring_time_mult": 3.0,  # very long ring
    },
    "wind_chime": {
        # Thin metal rods — delicate, high, shimmery
        "partials": [
            (1.0,   1.0,   4.0),
            (2.756, 0.50,  4.5),
            (5.404, 0.30,  5.0),
            (8.933, 0.15,  6.0),
            (13.34, 0.08,  8.0),
        ],
        "strike_brightness": 9000,
        "strike_amount": 0.08,
        "ring_time_mult": 0.5,
    },
}


def bell_strike(t, start, freq, duration, amplitude, profile_name, rng):
    """Synthesize a single bell strike with proper bell acoustics.

    Bell partials are NOT harmonic — they follow specific ratios
    determined by the bell's geometry and material. This is what
    makes a bell sound like a bell and not an organ.
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    profile = BELL_PROFILES[profile_name]

    if freq < 20 or freq > 10000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    signal = np.zeros(n_active)

    # --- Synthesize each bell partial ---
    for partial_ratio, partial_amp, decay_rate in profile["partials"]:
        partial_freq = freq * partial_ratio

        # Skip if above Nyquist
        if partial_freq > SAMPLE_RATE / 2 - 200:
            continue

        # Slight random detuning per partial (casting imperfections)
        detune = 1.0 + rng.uniform(-0.002, 0.002)
        partial_freq *= detune

        # Phase accumulation
        phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)

        # Exponential decay — each partial decays at its own rate
        # Multiply by ring_time_mult to scale overall ring duration
        effective_decay = decay_rate / profile["ring_time_mult"]
        envelope = np.exp(-effective_decay * t_local)

        # Slight amplitude beating between close partials (real bells do this)
        if partial_ratio > 1.0 and rng.random() < 0.4:
            beat_freq = rng.uniform(0.3, 2.0)
            beat_depth = rng.uniform(0.05, 0.15)
            envelope *= (1.0 + beat_depth * np.sin(2 * np.pi * beat_freq * t_local))

        signal += partial_amp * np.sin(phase) * envelope

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # --- Strike transient ---
    strike_noise = rng.randn(n_active)

    # Bandpass the noise around the strike brightness frequency
    center = profile["strike_brightness"]
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = 2.0
    alpha = np.sin(w0) / (2 * Q)
    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    strike_noise = lfilter(b, a, strike_noise)

    # Very fast decay — strike is < 10ms
    strike_len = min(int(0.008 * SAMPLE_RATE), n_active)
    strike_env = np.zeros(n_active)
    if strike_len > 0:
        strike_env[:strike_len] = np.exp(-np.linspace(0, 15, strike_len))

    strike = strike_noise * strike_env
    s_peak = np.max(np.abs(strike))
    if s_peak > 0:
        strike /= s_peak

    # Combine bell tone + strike
    combined = signal * 0.85 + strike * profile["strike_amount"]

    # Overall envelope — instant attack, natural ring decay
    overall_env = np.ones(n_active)
    # Tiny attack (1ms) to avoid click
    attack_len = max(int(0.001 * SAMPLE_RATE), 1)
    if attack_len < n_active:
        overall_env[:attack_len] = np.linspace(0, 1, attack_len)
    # Anti-click fadeout
    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        overall_env[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = combined * overall_env * amplitude
    return voice


# =====================================================================
# TICK-TOCK CLOCK — the Bergman metronome
# =====================================================================

def make_clock(t, bpm=60, amplitude=0.06):
    """Generate a relentless tick-tock clock.

    Two alternating percussive sounds:
      - TICK: higher, sharper (clock escapement forward)
      - TOCK: lower, slightly softer (escapement return)

    Like the sickroom clock in "Cries and Whispers" —
    you become aware it's been there the whole time.
    """
    audio = np.zeros_like(t)
    period = 60.0 / bpm  # seconds per beat

    # Tick and tock frequencies — wooden, resonant
    tick_freq = 3200   # higher click
    tock_freq = 2400   # lower knock

    n_beats = int(DURATION / period) + 1

    for i in range(n_beats):
        beat_time = i * period
        if beat_time >= DURATION - 0.1:
            break

        is_tick = (i % 2 == 0)
        freq = tick_freq if is_tick else tock_freq
        amp = amplitude if is_tick else amplitude * 0.75

        mask = (t >= beat_time) & (t < beat_time + 0.15)
        n_active = np.sum(mask)
        if n_active == 0:
            continue

        t_local = t[mask] - beat_time

        # Resonant click — narrow band noise with fast decay
        # Like a wooden pendulum mechanism
        click = np.sin(2 * np.pi * freq * t_local)

        # Add a body resonance at lower frequency
        body_freq = freq * 0.35
        body = 0.3 * np.sin(2 * np.pi * body_freq * t_local)

        combined = click + body

        # Very fast exponential decay — mechanical, not musical
        decay = np.exp(-t_local * 80)  # ~12ms effective duration

        # Tiny bit of randomness in timing (mechanical imperfection)
        # (already handled by the rigid beat_time — but we could add jitter)

        audio[mask] += combined * decay * amp

    return audio


def make_clock_resonance(t, bpm=60, amplitude=0.012):
    """Subtle sympathetic resonance from the clock mechanism.

    In a quiet room, you hear not just the tick but the case
    resonating, the pendulum's faint whoosh. This adds a ghostly
    undertone that makes the clock feel physical.
    """
    audio = np.zeros_like(t)
    period = 60.0 / bpm

    # Pendulum whoosh — very low frequency sine, synced to tick rate
    # A pendulum at 60 BPM swings at 0.5 Hz (one full swing per tick-tock pair)
    swing_freq = bpm / 120.0  # Hz
    pendulum = amplitude * np.sin(2 * np.pi * swing_freq * t)

    # Case resonance — a constant very quiet hum at the case's natural frequency
    case_freq = 180  # Hz — wooden case body resonance
    case_hum = amplitude * 0.3 * np.sin(2 * np.pi * case_freq * t)
    # Modulate by the pendulum swing
    case_hum *= (0.7 + 0.3 * np.abs(np.sin(2 * np.pi * swing_freq * t)))

    audio = pendulum + case_hum
    return audio


# =====================================================================
# BELL LAYERS — tonal, building
# =====================================================================

# D Dorian: D E F G A B C D
# Notes as pitch classes: D=2, E=4, F=5, G=7, A=9, B=11, C=0
D_DORIAN = [2, 4, 5, 7, 9, 11, 0]

# Closely related modes for shifts:
# A Mixolydian: A B C# D E F# G  → [9, 11, 1, 2, 4, 6, 7]
A_MIXOLYDIAN = [9, 11, 1, 2, 4, 6, 7]

# G Ionian: G A B C D E F# → [7, 9, 11, 0, 2, 4, 6]
G_MAJOR = [7, 9, 11, 0, 2, 4, 6]

# D minor (natural): D E F G A Bb C → [2, 4, 5, 7, 9, 10, 0]
D_MINOR = [2, 4, 5, 7, 9, 10, 0]


def freq_from_pc(pc, octave):
    """Convert pitch class + octave to frequency."""
    return BASE_FREQ * (2 ** ((pc - 0) / 12 + (octave - 4)))


def generate_bell_layer(t, events_out, rng, np_rng,
                        scale, octave_range, profile_name,
                        start_time, end_time,
                        density, amp_range, dur_range,
                        pattern="melodic"):
    """Generate a layer of bell events.

    Patterns:
      "melodic"  — stepwise/skip motion through the scale
      "arpeggic" — triadic leaps
      "ostinato" — repeating short figure
      "chordal"  — simultaneous notes
    """
    audio = np.zeros_like(t)
    window = end_time - start_time

    if pattern == "melodic":
        # Walk through the scale with occasional skips
        n_events = int(window * density)
        time_step = window / max(n_events, 1)
        scale_pos = rng.randint(0, len(scale) - 1)

        for i in range(n_events):
            ev_time = start_time + i * time_step + rng.uniform(-0.2, 0.2) * time_step
            ev_time = max(start_time, min(end_time - 1, ev_time))

            # Step or skip
            step = rng.choice([-2, -1, -1, 1, 1, 2])
            scale_pos = (scale_pos + step) % len(scale)
            pc = scale[scale_pos]

            octave = rng.choice(octave_range)
            dur = rng.uniform(*dur_range)
            amp = rng.uniform(*amp_range)

            audio += bell_strike(t, ev_time, freq_from_pc(pc, octave),
                               dur, amp, profile_name, np_rng)
            events_out.append({
                'time': ev_time, 'pc': pc, 'octave': octave,
                'dur': dur, 'amp': amp, 'profile': profile_name,
                'pattern': pattern
            })

    elif pattern == "arpeggic":
        # Triads broken upward/downward
        n_arpeggios = int(window * density / 3)
        time_step = window / max(n_arpeggios, 1)

        for i in range(n_arpeggios):
            base_time = start_time + i * time_step
            root_idx = rng.randint(0, len(scale) - 1)

            # Build a triad from scale degrees
            triad_indices = [root_idx, (root_idx + 2) % len(scale),
                           (root_idx + 4) % len(scale)]

            # Ascending or descending
            ascending = rng.random() < 0.6
            if not ascending:
                triad_indices.reverse()

            arp_delay = rng.uniform(0.08, 0.25)  # time between arp notes

            for j, idx in enumerate(triad_indices):
                pc = scale[idx]
                ev_time = base_time + j * arp_delay
                if ev_time >= end_time:
                    break

                octave = rng.choice(octave_range)
                # Higher notes in arpeggio slightly quieter
                amp = rng.uniform(*amp_range) * (1.0 - j * 0.1)
                dur = rng.uniform(*dur_range)

                audio += bell_strike(t, ev_time, freq_from_pc(pc, octave),
                                   dur, amp, profile_name, np_rng)
                events_out.append({
                    'time': ev_time, 'pc': pc, 'octave': octave,
                    'dur': dur, 'amp': amp, 'profile': profile_name,
                    'pattern': pattern
                })

    elif pattern == "ostinato":
        # Short repeating figure (3-5 notes)
        figure_len = rng.randint(3, 5)
        root_idx = rng.randint(0, len(scale) - 1)
        figure_pcs = [scale[(root_idx + k) % len(scale)]
                     for k in range(figure_len)]
        figure_octs = [rng.choice(octave_range) for _ in range(figure_len)]

        note_spacing = rng.uniform(0.4, 0.8)
        figure_dur = figure_len * note_spacing + rng.uniform(0.5, 1.5)  # gap between repeats

        current_time = start_time
        while current_time < end_time:
            for j in range(figure_len):
                ev_time = current_time + j * note_spacing
                if ev_time >= end_time:
                    break

                pc = figure_pcs[j]
                octave = figure_octs[j]
                amp = rng.uniform(*amp_range)
                dur = rng.uniform(*dur_range)

                audio += bell_strike(t, ev_time, freq_from_pc(pc, octave),
                                   dur, amp, profile_name, np_rng)
                events_out.append({
                    'time': ev_time, 'pc': pc, 'octave': octave,
                    'dur': dur, 'amp': amp, 'profile': profile_name,
                    'pattern': pattern
                })

            current_time += figure_dur

    elif pattern == "chordal":
        # Simultaneous notes — tower of bells
        n_chords = int(window * density / 4)
        time_step = window / max(n_chords, 1)

        for i in range(n_chords):
            chord_time = start_time + i * time_step + rng.uniform(-0.3, 0.3) * time_step
            chord_time = max(start_time, min(end_time - 2, chord_time))

            # 3-5 notes from scale
            n_notes = rng.randint(3, 5)
            root_idx = rng.randint(0, len(scale) - 1)

            for j in range(n_notes):
                idx = (root_idx + j * 2) % len(scale)  # stack thirds
                pc = scale[idx]
                octave = rng.choice(octave_range)
                # Slight timing spread (not perfectly simultaneous)
                ev_time = chord_time + rng.uniform(0, 0.04)
                amp = rng.uniform(*amp_range) * (0.9 ** j)  # inner voices quieter
                dur = rng.uniform(*dur_range)

                audio += bell_strike(t, ev_time, freq_from_pc(pc, octave),
                                   dur, amp, profile_name, np_rng)
                events_out.append({
                    'time': ev_time, 'pc': pc, 'octave': octave,
                    'dur': dur, 'amp': amp, 'profile': profile_name,
                    'pattern': pattern
                })

    return audio


# =====================================================================
# MAIN COMPOSITION
# =====================================================================

def generate_bells_piece(seed=None):
    """Generate the layered bells + tick-tock composition."""

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
    # LAYER 0: THE CLOCK (always present)
    # ===========================================
    print("  Building clock mechanism...")
    clock_bpm = 63  # slightly off 60 — more unsettling than metronomic
    clock = make_clock(t, bpm=clock_bpm, amplitude=0.055)
    clock_res = make_clock_resonance(t, bpm=clock_bpm, amplitude=0.008)
    clock_audio = clock + clock_res

    # The clock fades in from nothing (you become aware of it)
    clock_fade_in = np.minimum(t / 5.0, 1.0)
    # But never fades out — it's still ticking when the piece ends
    # (Actually, a very slight dip during the full peal, then back)
    clock_duck = np.ones_like(t)
    # Duck slightly during full peal (75-100s) — bells overpower the clock
    duck_mask = (t >= 75) & (t < 100)
    clock_duck[duck_mask] = 0.5
    # Smooth the ducking transitions
    smooth_len = int(3.0 * SAMPLE_RATE)
    duck_start = int(75 * SAMPLE_RATE)
    duck_end = int(100 * SAMPLE_RATE)
    if duck_start > smooth_len:
        ramp = np.linspace(1.0, 0.5, smooth_len)
        clock_duck[duck_start - smooth_len:duck_start] = ramp
    if duck_end + smooth_len < len(t):
        ramp = np.linspace(0.5, 1.0, smooth_len)
        clock_duck[duck_end:duck_end + smooth_len] = ramp

    clock_audio *= clock_fade_in * clock_duck

    # ===========================================
    # SECTION 1: THE CLOCK ALONE (0-15s)
    # ===========================================
    print("  Section 1: Clock alone...")
    # Just the clock — establishing dread. Maybe one distant bell at ~12s.
    audio += bell_strike(t, 12.0, freq_from_pc(2, 5), 6.0, 0.03,
                        "tubular_bell", np_rng)
    events.append({'time': 12.0, 'pc': 2, 'octave': 5, 'dur': 6.0,
                  'amp': 0.03, 'profile': 'tubular_bell', 'pattern': 'solo'})

    # ===========================================
    # SECTION 2: FIRST BELLS (15-40s)
    # ===========================================
    print("  Section 2: First bells (glockenspiel melody)...")

    # Single glockenspiel — D Dorian melody
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_DORIAN, octave_range=[5, 6],
        profile_name="glockenspiel",
        start_time=15, end_time=40,
        density=0.5,  # notes per second
        amp_range=(0.04, 0.07),
        dur_range=(1.5, 3.5),
        pattern="melodic"
    )

    # At 28s, a single celesta note answers — first hint of layering
    audio += bell_strike(t, 28.0, freq_from_pc(9, 4), 4.0, 0.04,
                        "celesta", np_rng)
    events.append({'time': 28.0, 'pc': 9, 'octave': 4, 'dur': 4.0,
                  'amp': 0.04, 'profile': 'celesta', 'pattern': 'answer'})

    # Wind chime accents (30-38s) — delicate
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_DORIAN, octave_range=[6, 7],
        profile_name="wind_chime",
        start_time=30, end_time=38,
        density=0.35,
        amp_range=(0.015, 0.03),
        dur_range=(0.8, 2.0),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 3: ACCUMULATION (40-75s)
    # ===========================================
    print("  Section 3: Accumulation...")

    # Glockenspiel continues — shift to A Mixolydian
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=A_MIXOLYDIAN, octave_range=[5, 6],
        profile_name="glockenspiel",
        start_time=40, end_time=60,
        density=0.6,
        amp_range=(0.04, 0.08),
        dur_range=(1.5, 3.0),
        pattern="melodic"
    )

    # Celesta ostinato enters (42-65s) — repeating figure
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=A_MIXOLYDIAN, octave_range=[4, 5],
        profile_name="celesta",
        start_time=42, end_time=65,
        density=0.45,
        amp_range=(0.03, 0.06),
        dur_range=(2.0, 4.0),
        pattern="ostinato"
    )

    # Tubular bells — deep tolling (50-75s)
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_MAJOR, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=50, end_time=75,
        density=0.18,
        amp_range=(0.05, 0.09),
        dur_range=(4.0, 8.0),
        pattern="arpeggic"
    )

    # Glockenspiel arpeggios (55-75s) — higher density
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_MAJOR, octave_range=[5, 6, 7],
        profile_name="glockenspiel",
        start_time=55, end_time=75,
        density=0.7,
        amp_range=(0.03, 0.06),
        dur_range=(1.0, 2.5),
        pattern="arpeggic"
    )

    # Wind chimes — continuous shimmer (55-75s)
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_MAJOR, octave_range=[6, 7],
        profile_name="wind_chime",
        start_time=55, end_time=75,
        density=0.5,
        amp_range=(0.01, 0.025),
        dur_range=(0.5, 1.5),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 4: FULL PEAL (75-100s)
    # ===========================================
    print("  Section 4: Full peal...")

    # D minor — coming home, but darker
    # Church bell enters — the deepest voice
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[2, 3],
        profile_name="church_bell",
        start_time=75, end_time=100,
        density=0.08,
        amp_range=(0.06, 0.10),
        dur_range=(6.0, 12.0),
        pattern="melodic"
    )

    # Tubular bells — chordal now
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=75, end_time=100,
        density=0.20,
        amp_range=(0.05, 0.09),
        dur_range=(4.0, 8.0),
        pattern="chordal"
    )

    # Celesta — arpeggic flourishes
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[4, 5, 6],
        profile_name="celesta",
        start_time=75, end_time=100,
        density=0.55,
        amp_range=(0.03, 0.06),
        dur_range=(2.0, 4.0),
        pattern="arpeggic"
    )

    # Glockenspiel — dense melodic shimmer
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[5, 6, 7],
        profile_name="glockenspiel",
        start_time=75, end_time=100,
        density=0.8,
        amp_range=(0.03, 0.06),
        dur_range=(1.0, 2.5),
        pattern="melodic"
    )

    # Wind chimes — continuous sparkle
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[6, 7],
        profile_name="wind_chime",
        start_time=78, end_time=98,
        density=0.7,
        amp_range=(0.01, 0.02),
        dur_range=(0.5, 1.5),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 5: DISSOLUTION (100-120s)
    # ===========================================
    print("  Section 5: Dissolution (clock remains)...")

    # Bells thin out — only celesta and occasional glockenspiel
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[5, 6],
        profile_name="celesta",
        start_time=100, end_time=112,
        density=0.25,
        amp_range=(0.025, 0.045),
        dur_range=(2.5, 5.0),
        pattern="melodic"
    )

    # Last glockenspiel notes — sparse, fading
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_MINOR, octave_range=[5, 6],
        profile_name="glockenspiel",
        start_time=105, end_time=115,
        density=0.15,
        amp_range=(0.02, 0.04),
        dur_range=(2.0, 4.0),
        pattern="melodic"
    )

    # One final deep bell at 112s — D2 tolling
    audio += bell_strike(t, 112.0, freq_from_pc(2, 2), 10.0, 0.06,
                        "church_bell", np_rng)
    events.append({'time': 112.0, 'pc': 2, 'octave': 2, 'dur': 10.0,
                  'amp': 0.06, 'profile': 'church_bell', 'pattern': 'final'})

    # ===========================================
    # MIX AND MASTER
    # ===========================================
    print("  Mixing and mastering...")

    # Add clock to bells
    audio += clock_audio

    # Reverb — generous, cathedral-like
    # Use longer delay times for more spacious reverb
    cathedral_delays = [31, 47, 67, 89, 113, 149, 191, 251, 313, 397]
    audio = simple_reverb(audio, decay=0.55, delays_ms=cathedral_delays,
                         sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 2.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 3.0, 1.0)
    # Don't fade out too much — the clock should persist
    fade_out = np.maximum(fade_out, 0.15)
    audio *= fade_in * fade_out

    # Normalize with generous headroom (bells are dynamic)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.15)

    # Gentle saturation — preserve bell transients
    audio = np.tanh(audio * 1.02) / 1.02

    return audio, events


def main():
    parser = argparse.ArgumentParser(
        description="Layered bells + Bergman tick-tock metronome"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="bells_bergman_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating bells + Bergman clock piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Structure: Clock Alone → First Bells → Accumulation → Full Peal → Dissolution")

    audio, events = generate_bells_piece(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")
    print(f"  Total bell events: {len(events)}")

    # Print section summary
    sections = {
        "Clock alone (0-15s)": [e for e in events if e['time'] < 15],
        "First bells (15-40s)": [e for e in events if 15 <= e['time'] < 40],
        "Accumulation (40-75s)": [e for e in events if 40 <= e['time'] < 75],
        "Full peal (75-100s)": [e for e in events if 75 <= e['time'] < 100],
        "Dissolution (100-120s)": [e for e in events if e['time'] >= 100],
    }
    for name, sec_events in sections.items():
        profiles = {}
        for e in sec_events:
            p = e['profile']
            profiles[p] = profiles.get(p, 0) + 1
        profile_str = ", ".join(f"{k}:{v}" for k, v in sorted(profiles.items()))
        print(f"  {name}: {len(sec_events)} events ({profile_str})")


if __name__ == "__main__":
    main()
