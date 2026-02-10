"""
cage_bells_family.py — Bells Family: Rude Guy, Boy, Papa & Little Sister

Four bell players collide in overlapping chaos:

  - RUDE GUY: Loops a fragment of the cage_ambient cluster interruption
    (~115-119s), pitch-shifted and rhythmically varied. He chews gum
    between hits and can't do everything at once. Sometimes fast,
    sometimes lazy, always impolite.

  - BOY: Church bell (F#5), clanging away with Bergman-film persistence.
    Irregular spacing, occasional double-hits for emphasis.

  - PAPA: Tubular bell (D2) with custom 10-partial bright profile —
    the bass foundation. Syncopated against the boy's clangs.
    Had schweinshaxe, playing from his gut. Double-whack 35% of the time.

  - LITTLE SISTER: Glockenspiel (B5), fastest of all. Giggly rapid-fire
    bursts, double-taps with slightly detuned second note, sloppy tuning.
    42 dings in 30 seconds. She's having fun.

The piece layers all four players additively — no damper pedal, all tails
ring into each other. Born from iterating on a 4-second slice of
cage_ambient's circle 4 (the cluster_rubber interruption at ~115s).

Duration: 30 seconds
Seed: 42

Usage:
    python cage_bells_family.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import random
import argparse
import json

from webern_pointillism import SAMPLE_RATE, simple_reverb, generate_noise

DURATION = 30
BASE_FREQ = 261.63  # C4


# =====================================================================
# BELL PROFILES
# =====================================================================

BELL_PROFILES = {
    "glockenspiel": {
        "partials": [
            (1.0,   1.0,   3.5),
            (2.76,  0.45,  4.0),
            (5.40,  0.25,  5.5),
            (8.93,  0.12,  7.0),
            (13.3,  0.06,  9.0),
        ],
        "strike_brightness": 8000,
        "strike_amount": 0.20,
        "ring_time_mult": 0.7,
    },
    "church_bell": {
        "partials": [
            (0.5,   0.35,  1.0),
            (1.0,   1.0,   1.2),
            (1.183, 0.80,  1.3),
            (1.506, 0.55,  1.5),
            (2.0,   0.65,  1.6),
            (2.514, 0.30,  2.0),
            (2.662, 0.22,  2.2),
            (3.011, 0.15,  2.5),
            (4.166, 0.08,  3.5),
            (5.433, 0.04,  4.5),
        ],
        "strike_brightness": 2000,
        "strike_amount": 0.18,
        "ring_time_mult": 3.0,
    },
    "tubular_bell": {
        "partials": [
            (0.5,   0.25,  1.5),
            (1.0,   1.0,   1.8),
            (1.183, 0.70,  2.0),
            (1.506, 0.45,  2.3),
            (2.0,   0.55,  2.5),
            (2.514, 0.20,  3.0),
            (3.011, 0.12,  3.5),
            (4.166, 0.05,  5.0),
        ],
        "strike_brightness": 3000,
        "strike_amount": 0.15,
        "ring_time_mult": 2.0,
    },
}

# Papa's custom 10-partial bright profile — extra upper modes for audibility
PAPA_BELL = {
    "partials": [
        (0.5,   0.30,  1.2),
        (1.0,   1.0,   1.5),
        (1.183, 0.70,  1.8),
        (1.506, 0.50,  2.0),
        (2.0,   0.65,  2.0),
        (2.514, 0.40,  2.5),
        (2.662, 0.35,  2.8),
        (3.011, 0.30,  3.0),
        (4.0,   0.20,  3.5),
        (5.0,   0.12,  4.0),
    ],
    "strike_brightness": 3500,
    "strike_amount": 0.12,
    "ring_time_mult": 2.5,
}


# =====================================================================
# BELL SYNTHESIS
# =====================================================================

def bell_strike(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a single bell strike with proper bell acoustics.

    Bell partials are NOT harmonic — they follow specific ratios
    determined by the bell's geometry and material.

    profile can be a string (key into BELL_PROFILES) or a dict.
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)

    if isinstance(profile, str):
        profile = BELL_PROFILES[profile]

    if freq < 20 or freq > 10000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = int(np.sum(mask))
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    signal = np.zeros(n_active)

    for partial_ratio, partial_amp, decay_rate in profile["partials"]:
        partial_freq = freq * partial_ratio
        if partial_freq > SAMPLE_RATE / 2 - 200:
            continue

        detune = 1.0 + rng.uniform(-0.002, 0.002)
        partial_freq *= detune
        phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)

        effective_decay = decay_rate / profile["ring_time_mult"]
        envelope = np.exp(-effective_decay * t_local)

        if partial_ratio > 1.0 and rng.random() < 0.4:
            beat_freq = rng.uniform(0.3, 2.0)
            beat_depth = rng.uniform(0.05, 0.15)
            envelope *= (1.0 + beat_depth * np.sin(2 * np.pi * beat_freq * t_local))

        signal += partial_amp * np.sin(phase) * envelope

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # Strike transient
    strike_noise = rng.randn(n_active)
    center = profile["strike_brightness"]
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = 2.0
    alpha = np.sin(w0) / (2 * Q)
    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    strike_noise = lfilter(b, a, strike_noise)

    strike_len = min(int(0.008 * SAMPLE_RATE), n_active)
    strike_env = np.zeros(n_active)
    if strike_len > 0:
        strike_env[:strike_len] = np.exp(-np.linspace(0, 15, strike_len))

    strike = strike_noise * strike_env
    s_peak = np.max(np.abs(strike))
    if s_peak > 0:
        strike /= s_peak

    combined = signal * 0.85 + strike * profile["strike_amount"]

    overall_env = np.ones(n_active)
    attack_len = max(int(0.001 * SAMPLE_RATE), 1)
    if attack_len < n_active:
        overall_env[:attack_len] = np.linspace(0, 1, attack_len)
    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        overall_env[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = combined * overall_env * amplitude
    return voice


# =====================================================================
# RUDE GUY — cage_ambient fragment looper
# =====================================================================

def load_cage_fragment():
    """Load the 115-119s slice from cage_ambient_01.wav."""
    try:
        sr, data = wavfile.read('cage_ambient_01.wav')
    except FileNotFoundError:
        # Fallback: synthesize a rough prepared piano cluster
        return synthesize_cluster_fragment()

    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0

    if data.ndim > 1:
        data = data.mean(axis=1)

    # Extract 115s to 119s (the circle 4 cluster)
    start_sample = int(115 * sr)
    end_sample = int(119 * sr)
    fragment = data[start_sample:end_sample]

    # Resample if needed
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n_new = int(len(fragment) * ratio)
        x_old = np.linspace(0, 1, len(fragment))
        x_new = np.linspace(0, 1, n_new)
        fragment = np.interp(x_new, x_old, fragment)

    return fragment


def synthesize_cluster_fragment():
    """Fallback: synthesize a rough prepared piano cluster if WAV not found."""
    rng = np.random.RandomState(42)
    sr = SAMPLE_RATE
    dur = 4.0
    n = int(dur * sr)
    t = np.linspace(0, dur, n)
    sig = np.zeros(n)

    # Cluster of 8-12 notes, prepared piano style
    n_notes = rng.randint(8, 13)
    for _ in range(n_notes):
        freq = 200 * (2 ** (rng.uniform(-1, 2)))
        amp = rng.uniform(0.05, 0.2)
        # Inharmonic partials
        for j in range(rng.randint(3, 7)):
            pf = freq * (j + 1) * (1.0 + rng.uniform(-0.06, 0.06))
            if pf > sr / 2 - 200:
                continue
            env = np.exp(-rng.uniform(2, 8) * t)
            sig += amp * 0.3 * np.sin(2 * np.pi * pf * t + rng.uniform(0, 6.28)) * env

    # Strike noise
    noise = rng.randn(n) * np.exp(-20 * t) * 0.3
    sig += noise

    peak = np.max(np.abs(sig))
    if peak > 0:
        sig /= peak
    return sig * 0.6


def rude_guy_layer(t, fragment, rng, events_list):
    """One rude guy with soul — varies pitch, plays fragments, rhythm variety.

    He mixes up his riff: sometimes quick, sometimes cruising, sometimes lazy.
    Can't do everything at once because he's chewing gum.
    """
    audio = np.zeros(len(t))
    frag_len = len(fragment)

    # Schedule: irregular hits with rhythm variety
    current_time = rng.uniform(0.3, 0.8)
    hit_count = 0

    while current_time < DURATION - 1.0:
        # Rhythm mode: quick / cruising / lazy
        mode = rng.choice(['quick', 'cruising', 'lazy'], p=[0.25, 0.5, 0.25])

        # Pitch shift this loop (soul = variation)
        pitch_shift = rng.uniform(0.93, 1.03)

        # Sometimes play a fragment, sometimes the full 4s chunk
        if rng.random() < 0.4:
            # Fragment — just 1-2.5 seconds
            frag_dur = rng.uniform(1.0, 2.5)
            frag_start = rng.randint(0, max(1, frag_len - int(frag_dur * SAMPLE_RATE)))
            n_use = min(int(frag_dur * SAMPLE_RATE), frag_len - frag_start)
        else:
            frag_start = 0
            n_use = frag_len

        # Pitch shift by resampling
        n_shifted = int(n_use / pitch_shift)
        chunk = fragment[frag_start:frag_start + n_use]
        x_old = np.linspace(0, 1, len(chunk))
        x_new = np.linspace(0, 1, n_shifted)
        shifted = np.interp(x_new, x_old, chunk)

        # Place in audio
        start_idx = int(current_time * SAMPLE_RATE)
        end_idx = min(start_idx + len(shifted), len(audio))
        actual_len = end_idx - start_idx
        if actual_len > 0:
            # Fade edges
            fade = min(int(0.01 * SAMPLE_RATE), actual_len)
            env = np.ones(actual_len)
            if fade > 0:
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] *= np.linspace(1, 0, fade)
            audio[start_idx:end_idx] += shifted[:actual_len] * env * 0.35

            # Track event
            pc = int(pitch_shift * 12) % 12
            events_list.append({
                'time': float(current_time),
                'pc': pc,
                'octave': 4,
                'duration': float(actual_len / SAMPLE_RATE),
                'amplitude': 0.35,
                'type': 'cluster',
                'category': 'rude_guy'
            })

        # Gap depends on mode
        if mode == 'quick':
            gap = rng.uniform(1.5, 2.5)
        elif mode == 'cruising':
            gap = rng.uniform(3.0, 4.5)
        else:  # lazy
            gap = rng.uniform(5.0, 7.0)

        current_time += len(shifted) / SAMPLE_RATE + gap
        hit_count += 1

    return audio


# =====================================================================
# BOY — church bell F#5
# =====================================================================

def boy_layer(t, rng, events_list):
    """Boy with F#5 church bell — 18 clangs, irregular timing, double-hits."""
    audio = np.zeros(len(t))
    boy_freq = 739.99  # F#5

    current_time = rng.uniform(0.5, 1.2)
    clang_count = 0
    max_clangs = 18

    while clang_count < max_clangs and current_time < DURATION - 0.5:
        amp = rng.uniform(0.12, 0.22)
        dur = rng.uniform(2.5, 4.0)

        audio += bell_strike(t, current_time, boy_freq, dur, amp,
                            "church_bell", rng)

        events_list.append({
            'time': float(current_time),
            'pc': 6,  # F#
            'octave': 5,
            'duration': float(dur),
            'amplitude': float(amp),
            'type': 'church_bell',
            'category': 'boy'
        })
        clang_count += 1

        # Double hit?
        if rng.random() < 0.2 and clang_count < max_clangs:
            double_gap = rng.uniform(0.08, 0.15)
            current_time += double_gap
            amp2 = amp * rng.uniform(0.6, 0.9)
            audio += bell_strike(t, current_time, boy_freq, dur * 0.8, amp2,
                                "church_bell", rng)

            events_list.append({
                'time': float(current_time),
                'pc': 6,
                'octave': 5,
                'duration': float(dur * 0.8),
                'amplitude': float(amp2),
                'type': 'church_bell',
                'category': 'boy'
            })
            clang_count += 1

        # Gap
        gap = rng.uniform(1.0, 2.5)
        current_time += gap

    return audio


# =====================================================================
# PAPA — D2 with custom bright 10-partial profile
# =====================================================================

def papa_layer(t, rng, events_list):
    """Papa on D2 tubular bell — syncopated, loud, double-whack 35%."""
    audio = np.zeros(len(t))
    papa_freq = 73.42  # D2

    current_time = rng.uniform(1.0, 2.0)
    toll_count = 0
    max_tolls = 22

    while toll_count < max_tolls and current_time < DURATION - 0.5:
        amp = rng.uniform(0.25, 0.40)
        dur = rng.uniform(3.5, 5.0)

        audio += bell_strike(t, current_time, papa_freq, dur, amp,
                            PAPA_BELL, rng)

        events_list.append({
            'time': float(current_time),
            'pc': 2,  # D
            'octave': 2,
            'duration': float(dur),
            'amplitude': float(amp),
            'type': 'tubular_bell',
            'category': 'papa'
        })
        toll_count += 1

        # Double whack? 35% chance
        if rng.random() < 0.35 and toll_count < max_tolls:
            double_gap = rng.uniform(0.12, 0.25)
            current_time += double_gap
            amp2 = amp * rng.uniform(0.7, 1.0)
            audio += bell_strike(t, current_time, papa_freq, dur * 0.9, amp2,
                                PAPA_BELL, rng)

            events_list.append({
                'time': float(current_time),
                'pc': 2,
                'octave': 2,
                'duration': float(dur * 0.9),
                'amplitude': float(amp2),
                'type': 'tubular_bell',
                'category': 'papa'
            })
            toll_count += 1

        # Gap — syncopated (offset from boy's rhythm)
        gap = rng.uniform(0.8, 2.0)
        current_time += gap

    return audio


# =====================================================================
# LITTLE SISTER — B5 glockenspiel
# =====================================================================

def sister_layer(t, rng, events_list):
    """Little sister on B5 glockenspiel — 42 dings, rapid-fire, giggly."""
    audio = np.zeros(len(t))
    sister_freq = 987.77  # B5

    current_time = rng.uniform(0.3, 0.7)
    ding_count = 0
    max_dings = 42

    while ding_count < max_dings and current_time < DURATION - 0.3:
        amp = rng.uniform(0.08, 0.18)
        dur = rng.uniform(1.5, 2.5)

        # Sloppy tuning — sister doesn't care
        sloppy = sister_freq * rng.uniform(0.98, 1.02)

        audio += bell_strike(t, current_time, sloppy, dur, amp,
                            "glockenspiel", rng)

        # Determine pitch class from sloppy freq
        pc = 11  # B
        events_list.append({
            'time': float(current_time),
            'pc': pc,
            'octave': 5,
            'duration': float(dur),
            'amplitude': float(amp),
            'type': 'glockenspiel',
            'category': 'sister'
        })
        ding_count += 1

        # Rapid-fire burst? 25% chance
        if rng.random() < 0.25 and ding_count < max_dings - 2:
            burst = rng.randint(2, 4)
            for _ in range(burst):
                burst_gap = rng.uniform(0.15, 0.4)
                current_time += burst_gap
                if current_time >= DURATION - 0.3:
                    break
                burst_amp = amp * rng.uniform(0.7, 1.1)
                burst_sloppy = sister_freq * rng.uniform(0.97, 1.03)

                audio += bell_strike(t, current_time, burst_sloppy, dur * 0.7,
                                    burst_amp, "glockenspiel", rng)

                events_list.append({
                    'time': float(current_time),
                    'pc': pc,
                    'octave': 5,
                    'duration': float(dur * 0.7),
                    'amplitude': float(burst_amp),
                    'type': 'glockenspiel',
                    'category': 'sister'
                })
                ding_count += 1

        # Double-tap? 20% chance with detuned second note
        elif rng.random() < 0.20 and ding_count < max_dings:
            tap_gap = rng.uniform(0.06, 0.12)
            current_time += tap_gap
            detune = rng.choice([1.06, 0.94])
            tap_freq = sister_freq * detune
            tap_amp = amp * rng.uniform(0.5, 0.8)

            audio += bell_strike(t, current_time, tap_freq, dur * 0.6,
                                tap_amp, "glockenspiel", rng)

            events_list.append({
                'time': float(current_time),
                'pc': pc,
                'octave': 5,
                'duration': float(dur * 0.6),
                'amplitude': float(tap_amp),
                'type': 'glockenspiel',
                'category': 'sister'
            })
            ding_count += 1

        # Gap — she's fast
        gap = rng.uniform(0.3, 1.2)
        current_time += gap

    return audio


# =====================================================================
# MAIN COMPOSITION
# =====================================================================

def generate(seed=42, output='cage_bells_family_01.wav'):
    rng = np.random.RandomState(seed)
    random.seed(seed)

    n_samples = int(DURATION * SAMPLE_RATE)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)
    audio = np.zeros(n_samples)
    events = []

    print("Loading cage_ambient fragment (115-119s)...")
    fragment = load_cage_fragment()
    print(f"  Fragment: {len(fragment)} samples ({len(fragment)/SAMPLE_RATE:.1f}s)")

    print("Generating rude guy...")
    audio += rude_guy_layer(t, fragment, rng, events)

    print("Generating boy (F#5 church bell)...")
    audio += boy_layer(t, rng, events)

    print("Generating papa (D2 bright tubular)...")
    audio += papa_layer(t, rng, events)

    print("Generating little sister (B5 glockenspiel)...")
    audio += sister_layer(t, rng, events)

    # Light reverb — room, not cathedral
    room_delays = [17, 29, 43, 61, 79]
    audio = simple_reverb(audio, decay=0.3, delays_ms=room_delays,
                         sample_rate=SAMPLE_RATE)

    # Gentle fadeout at the end (let tails ring)
    fade_len = int(2.0 * SAMPLE_RATE)
    if fade_len < len(audio):
        audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.92

    # Sort events by time
    events.sort(key=lambda e: e['time'])

    print(f"\nTotal events: {len(events)}")
    print(f"  Rude guy: {sum(1 for e in events if e['category'] == 'rude_guy')}")
    print(f"  Boy:      {sum(1 for e in events if e['category'] == 'boy')}")
    print(f"  Papa:     {sum(1 for e in events if e['category'] == 'papa')}")
    print(f"  Sister:   {sum(1 for e in events if e['category'] == 'sister')}")

    # Write WAV
    audio_16 = np.int16(audio * 32767)
    wavfile.write(output, SAMPLE_RATE, audio_16)
    print(f"\nWritten: {output}")

    # Write JSON for animated player
    json_file = output.replace('.wav', '').replace('_01', '') + '.json'
    if json_file.endswith('.json'):
        pass
    else:
        json_file = 'cage_bells_family.json'

    json_data = {
        'title': 'Bells Family — Rude Guy, Boy, Papa & Little Sister',
        'duration': DURATION,
        'seed': seed,
        'audio_file': output.replace('.wav', '.mp3'),
        'events': events
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Written: {json_file}")

    return audio, events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bells Family composition')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='cage_bells_family_01.wav')
    args = parser.parse_args()

    generate(seed=args.seed, output=args.output)
