"""
bells_gentle.py — Gentle bells for Tanya

Rules:
  - No sudden loud anything
  - No metallic/toy sounds
  - Pizzicato rhythm guides the mind
  - Wood xylophone (rosewood bars over resonator tubes)
  - Warm, soft, continuous — like a music box remembering a lullaby

Timbres:
  - Pizzicato clock (from bells_pizz — the keeper)
  - Wood xylophone (NEW — rosewood bars, warm, rounded attack)
  - Celesta (soft, crystalline)
  - Tubular bells (QUIET — distant, cathedral)
  - Marimba-like low warmth (wood resonance)

Everything stays pp-mp. No ff. No surprises. The loudest moment
should feel like a warm swell, not a shock.

Usage:
    python bells_gentle.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import random
import argparse

from webern_pointillism import (
    SAMPLE_RATE, TIMBRES, pointillist_tone, simple_reverb
)
from bells_bergman import bell_strike, freq_from_pc, generate_bell_layer

DURATION = 120
BASE_FREQ = 261.63

# Scales — warm, gentle modes
D_DORIAN = [2, 4, 5, 7, 9, 11, 0]       # D E F G A B C
G_LYDIAN = [7, 9, 11, 1, 2, 4, 5]        # G A B C# D E F — dreamy, bright
A_PENT = [9, 0, 2, 4, 7]                  # A C D E G — pentatonic, no tension
F_MAJOR = [5, 7, 9, 10, 0, 2, 4]         # F G A Bb C D E — warm


# =====================================================================
# WOOD XYLOPHONE — rosewood bars over resonator tubes
# =====================================================================

# Real xylophones (rosewood bars) have:
#   - Fundamental + octave partial (bar tuned to emphasize 4:1 mode)
#   - Quick attack from hard mallet on wood
#   - Short sustain (wood absorbs energy)
#   - Warm, round tone — NOT metallic
#   - Resonator tube adds body at fundamental

WOOD_XYLO_PROFILE = {
    "partials": [
        # (ratio, amplitude, decay_rate)
        (1.0,   1.0,   3.0),     # fundamental — strong, moderate decay
        (3.0,   0.15,  5.0),     # 3rd partial (weak — wood dampens odd modes)
        (4.0,   0.40,  4.0),     # tuned octave-double (the bright "wood" sound)
        (6.27,  0.06,  7.0),     # higher mode — very quiet
    ],
    "strike_brightness": 4000,    # mallet on wood — warm, not harsh
    "strike_amount": 0.10,        # gentle strike
    "ring_time_mult": 0.6,        # shorter ring than metal
}

# Marimba — deeper, warmer cousin (rubber mallets on rosewood)
MARIMBA_PROFILE = {
    "partials": [
        (1.0,   1.0,   2.0),     # fundamental — sustained by resonator
        (4.0,   0.25,  3.5),     # tuned octave-double
        (2.8,   0.05,  5.0),     # very faint sub-mode
    ],
    "strike_brightness": 2000,    # rubber mallet — very warm
    "strike_amount": 0.06,        # soft attack
    "ring_time_mult": 1.0,        # resonator tube sustains
}


def wood_strike(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a wooden bar being struck.

    Similar structure to bell_strike but with wood characteristics:
      - Warmer attack (no metallic ring)
      - Shorter natural sustain
      - Resonator tube body underneath
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)

    if freq < 20 or freq > 8000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    signal = np.zeros(n_active)

    for partial_ratio, partial_amp, decay_rate in profile["partials"]:
        partial_freq = freq * partial_ratio
        if partial_freq > SAMPLE_RATE / 2 - 200:
            continue

        # Slight detuning (wood grain irregularities)
        detune = 1.0 + rng.uniform(-0.001, 0.001)
        partial_freq *= detune

        phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)

        effective_decay = decay_rate / profile["ring_time_mult"]
        envelope = np.exp(-effective_decay * t_local)

        signal += partial_amp * np.sin(phase) * envelope

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # Wood strike — filtered noise, warm bandpass
    strike_noise = rng.randn(n_active)
    center = profile["strike_brightness"]
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = 1.5  # wider Q than bells — softer attack
    alpha = np.sin(w0) / (2 * Q)
    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    strike_noise = lfilter(b, a, strike_noise)

    # Fast but not instant decay — wood has a "thud" not a "click"
    strike_len = min(int(0.012 * SAMPLE_RATE), n_active)
    strike_env = np.zeros(n_active)
    if strike_len > 0:
        strike_env[:strike_len] = np.exp(-np.linspace(0, 10, strike_len))

    strike = strike_noise * strike_env
    s_peak = np.max(np.abs(strike))
    if s_peak > 0:
        strike /= s_peak

    combined = signal * 0.88 + strike * profile["strike_amount"]

    # Resonator body hum — low sine at fundamental, fading slowly
    res_freq = freq
    res_phase = 2 * np.pi * res_freq * t_local + rng.uniform(0, 2 * np.pi)
    res_signal = 0.08 * np.sin(res_phase) * np.exp(-1.5 * t_local)
    combined += res_signal

    # Envelope
    overall_env = np.ones(n_active)
    attack_len = max(int(0.003 * SAMPLE_RATE), 1)  # 3ms — softer than bell
    if attack_len < n_active:
        overall_env[:attack_len] = np.linspace(0, 1, attack_len)
    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        overall_env[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = combined * overall_env * amplitude
    return voice


def generate_wood_layer(t, events_out, rng, np_rng,
                        scale, octave_range, profile,
                        start_time, end_time,
                        density, amp_range, dur_range,
                        pattern="melodic"):
    """Generate a layer of wood xylophone/marimba events."""
    audio = np.zeros_like(t)
    window = end_time - start_time

    if pattern == "melodic":
        n_events = int(window * density)
        time_step = window / max(n_events, 1)
        scale_pos = rng.randint(0, len(scale) - 1)

        for i in range(n_events):
            ev_time = start_time + i * time_step + rng.uniform(-0.15, 0.15) * time_step
            ev_time = max(start_time, min(end_time - 0.5, ev_time))

            step = rng.choice([-2, -1, -1, 1, 1, 2])
            scale_pos = (scale_pos + step) % len(scale)
            pc = scale[scale_pos]

            octave = rng.choice(octave_range)
            dur = rng.uniform(*dur_range)
            amp = rng.uniform(*amp_range)

            audio += wood_strike(t, ev_time, freq_from_pc(pc, octave),
                                dur, amp, profile, np_rng)
            events_out.append({
                'time': ev_time, 'pc': pc, 'octave': octave,
                'dur': dur, 'amp': amp, 'profile': 'xylophone' if profile == WOOD_XYLO_PROFILE else 'marimba',
                'pattern': pattern
            })

    elif pattern == "arpeggic":
        n_arps = int(window * density / 3)
        time_step = window / max(n_arps, 1)

        for i in range(n_arps):
            base_time = start_time + i * time_step
            root_idx = rng.randint(0, len(scale) - 1)
            triad = [root_idx, (root_idx + 2) % len(scale),
                    (root_idx + 4) % len(scale)]

            if rng.random() < 0.4:
                triad.reverse()

            arp_delay = rng.uniform(0.12, 0.30)

            for j, idx in enumerate(triad):
                pc = scale[idx]
                ev_time = base_time + j * arp_delay
                if ev_time >= end_time:
                    break

                octave = rng.choice(octave_range)
                amp = rng.uniform(*amp_range) * (1.0 - j * 0.08)
                dur = rng.uniform(*dur_range)

                audio += wood_strike(t, ev_time, freq_from_pc(pc, octave),
                                    dur, amp, profile, np_rng)
                events_out.append({
                    'time': ev_time, 'pc': pc, 'octave': octave,
                    'dur': dur, 'amp': amp, 'profile': 'xylophone' if profile == WOOD_XYLO_PROFILE else 'marimba',
                    'pattern': pattern
                })

    return audio


# =====================================================================
# PIZZICATO CLOCK — from bells_pizz, but gentler
# =====================================================================

def make_gentle_pizz(t, rng, np_rng, bpm=66, amplitude=0.045):
    """Gentle pizzicato clock — slower, softer than bells_pizz."""
    audio = np.zeros_like(t)
    pizz = TIMBRES[6]  # pizzicato timbre
    period = 60.0 / bpm

    tick_pc, tick_oct = 9, 2    # A2
    tock_pc, tock_oct = 4, 3    # E3

    n_beats = int(DURATION / period) + 1
    events = []

    for i in range(n_beats):
        beat_time = i * period
        if beat_time >= DURATION - 0.5:
            break

        beat_in_bar = i % 8

        if beat_in_bar == 0:
            audio += pointillist_tone(
                t, beat_time, tick_pc, tick_oct,
                0.9, amplitude * 1.0, pizz, np_rng
            )
            events.append({'time': beat_time, 'type': 'tick'})

        elif beat_in_bar == 4:
            audio += pointillist_tone(
                t, beat_time, tock_pc, tock_oct - 1,
                0.7, amplitude * 0.75, pizz, np_rng
            )
            events.append({'time': beat_time, 'type': 'tock'})

        elif beat_in_bar == 3 or beat_in_bar == 7:
            # Two-note pickup
            pickup_pc = rng.choice(A_PENT)
            audio += pointillist_tone(
                t, beat_time, pickup_pc, 3,
                0.4, amplitude * 0.4, pizz, np_rng
            )
            audio += pointillist_tone(
                t, beat_time + period * 0.5, rng.choice(A_PENT), 3,
                0.3, amplitude * 0.35, pizz, np_rng
            )
            events.append({'time': beat_time, 'type': 'figure'})

        elif beat_in_bar == 6:
            # Three-note turn
            turn_root = rng.choice(A_PENT)
            for j, delay in enumerate([0, 0.12, 0.24]):
                pc = A_PENT[(A_PENT.index(turn_root) + j) % len(A_PENT)]
                audio += pointillist_tone(
                    t, beat_time + delay, pc, 3,
                    0.35, amplitude * 0.38 * (0.9 ** j), pizz, np_rng
                )
            events.append({'time': beat_time, 'type': 'turn'})

        else:
            is_tick = (i % 2 == 0)
            pc = tick_pc if is_tick else tock_pc
            oct = tick_oct if is_tick else tock_oct
            amp = amplitude * (0.6 if is_tick else 0.45)

            audio += pointillist_tone(
                t, beat_time, pc, oct,
                0.5, amp, pizz, np_rng
            )
            events.append({'time': beat_time, 'type': 'tick' if is_tick else 'tock'})

    return audio, events


# =====================================================================
# MAIN COMPOSITION
# =====================================================================

def generate_gentle_bells(seed=None):
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
    # PIZZICATO HEARTBEAT
    # ===========================================
    print("  Building pizzicato heartbeat...")
    pizz_audio, _ = make_gentle_pizz(t, rng, np_rng, bpm=66, amplitude=0.045)
    pizz_fade = np.minimum(t / 4.0, 1.0)
    pizz_audio *= pizz_fade

    # ===========================================
    # SECTION 1: PIZZ + MARIMBA (0-25s)
    # ===========================================
    print("  Section 1: Pizz + marimba warmth (0-25s)")

    # Low marimba — warm foundation
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=A_PENT, octave_range=[3],
        profile=MARIMBA_PROFILE,
        start_time=5, end_time=25,
        density=0.2,
        amp_range=(0.03, 0.05),
        dur_range=(1.5, 3.0),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 2: XYLOPHONE ENTERS (20-50s)
    # ===========================================
    print("  Section 2: Wood xylophone melody (20-50s)")

    # Xylophone — D Dorian melody
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=D_DORIAN, octave_range=[4, 5],
        profile=WOOD_XYLO_PROFILE,
        start_time=20, end_time=50,
        density=0.4,
        amp_range=(0.04, 0.065),
        dur_range=(1.0, 2.0),
        pattern="melodic"
    )

    # Marimba continues underneath
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=D_DORIAN, octave_range=[3, 4],
        profile=MARIMBA_PROFILE,
        start_time=22, end_time=48,
        density=0.18,
        amp_range=(0.025, 0.04),
        dur_range=(2.0, 3.5),
        pattern="arpeggic"
    )

    # Celesta — distant, sparkly
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=D_DORIAN, octave_range=[5, 6],
        profile_name="celesta",
        start_time=30, end_time=48,
        density=0.2,
        amp_range=(0.02, 0.035),
        dur_range=(2.0, 4.0),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 3: FULL WARMTH (45-80s)
    # ===========================================
    print("  Section 3: Full warmth — layered wood + soft bells (45-80s)")

    # Xylophone arpeggios — G Lydian (dreamy)
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=G_LYDIAN, octave_range=[4, 5],
        profile=WOOD_XYLO_PROFILE,
        start_time=45, end_time=75,
        density=0.45,
        amp_range=(0.04, 0.06),
        dur_range=(1.0, 2.0),
        pattern="arpeggic"
    )

    # Marimba melodic line
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=G_LYDIAN, octave_range=[3, 4],
        profile=MARIMBA_PROFILE,
        start_time=48, end_time=78,
        density=0.25,
        amp_range=(0.03, 0.05),
        dur_range=(1.5, 3.0),
        pattern="melodic"
    )

    # Distant tubular bells — QUIET, like a memory
    # Max amplitude 0.04 — whisper level
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_LYDIAN, octave_range=[3, 4],
        profile_name="tubular_bell",
        start_time=55, end_time=75,
        density=0.06,   # very sparse
        amp_range=(0.02, 0.04),  # quiet!
        dur_range=(5.0, 8.0),
        pattern="melodic"
    )

    # Celesta arpeggios
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_LYDIAN, octave_range=[5, 6],
        profile_name="celesta",
        start_time=50, end_time=78,
        density=0.3,
        amp_range=(0.02, 0.04),
        dur_range=(2.0, 3.5),
        pattern="arpeggic"
    )

    # Wind chimes — barely there, high shimmer
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=G_LYDIAN, octave_range=[6, 7],
        profile_name="wind_chime",
        start_time=55, end_time=75,
        density=0.2,
        amp_range=(0.008, 0.015),
        dur_range=(0.5, 1.5),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 4: GENTLE DESCENT (75-105s)
    # ===========================================
    print("  Section 4: Gentle descent (75-105s)")

    # Xylophone — F major (warm resolution)
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=F_MAJOR, octave_range=[4, 5],
        profile=WOOD_XYLO_PROFILE,
        start_time=75, end_time=100,
        density=0.3,
        amp_range=(0.03, 0.05),
        dur_range=(1.2, 2.5),
        pattern="melodic"
    )

    # Marimba — slower, deeper
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=F_MAJOR, octave_range=[3],
        profile=MARIMBA_PROFILE,
        start_time=78, end_time=102,
        density=0.15,
        amp_range=(0.025, 0.04),
        dur_range=(2.0, 4.0),
        pattern="melodic"
    )

    # One distant tubular bell — F3, pppp
    audio += bell_strike(t, 90.0, freq_from_pc(5, 3), 8.0, 0.03,
                        "tubular_bell", np_rng)
    events.append({'time': 90.0, 'pc': 5, 'octave': 3, 'dur': 8.0,
                  'amp': 0.03, 'profile': 'tubular_bell', 'pattern': 'distant'})

    # Celesta — fading
    audio += generate_bell_layer(
        t, events, rng, np_rng,
        scale=F_MAJOR, octave_range=[5, 6],
        profile_name="celesta",
        start_time=80, end_time=100,
        density=0.15,
        amp_range=(0.015, 0.03),
        dur_range=(2.5, 4.0),
        pattern="melodic"
    )

    # ===========================================
    # SECTION 5: JUST BREATHING (100-120s)
    # ===========================================
    print("  Section 5: Just breathing (100-120s)")

    # Only xylophone and pizz remain
    audio += generate_wood_layer(
        t, events, rng, np_rng,
        scale=A_PENT, octave_range=[4, 5],
        profile=WOOD_XYLO_PROFILE,
        start_time=100, end_time=115,
        density=0.15,
        amp_range=(0.02, 0.035),
        dur_range=(1.5, 3.0),
        pattern="melodic"
    )

    # Last marimba note — A3
    audio += wood_strike(t, 112.0, freq_from_pc(9, 3), 5.0, 0.03,
                        MARIMBA_PROFILE, np_rng)
    events.append({'time': 112.0, 'pc': 9, 'octave': 3, 'dur': 5.0,
                  'amp': 0.03, 'profile': 'marimba', 'pattern': 'final'})

    # ===========================================
    # MIX
    # ===========================================
    print("  Mixing...")
    audio += pizz_audio

    # Warm reverb — slightly less cathedral, more intimate room
    room_delays = [29, 43, 61, 83, 107, 139, 181, 233]
    audio = simple_reverb(audio, decay=0.45, delays_ms=room_delays,
                         sample_rate=SAMPLE_RATE)

    # Gentle global envelope
    fade_in = np.minimum(t / 3.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 5.0, 1.0)
    fade_out = np.maximum(fade_out, 0.08)
    audio *= fade_in * fade_out

    # Normalize — extra headroom (stay soft)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.25)  # generous headroom

    # Very light saturation
    audio = np.tanh(audio * 1.01) / 1.01

    return audio, events


def main():
    parser = argparse.ArgumentParser(
        description="Gentle bells + wood xylophone — for Tanya"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="bells_gentle_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating gentle bells piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Max dynamic: mp (no surprises)\n")

    audio, events = generate_gentle_bells(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")

    profiles = {}
    for e in events:
        p = e.get('profile', '?')
        profiles[p] = profiles.get(p, 0) + 1
    for p, c in sorted(profiles.items()):
        print(f"    {p}: {c}")


if __name__ == "__main__":
    main()
