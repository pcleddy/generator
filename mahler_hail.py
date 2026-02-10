#!/usr/bin/env python3
"""
mahler_hail.py — "Hagelstürme" (Hailstorms)
=============================================
A Mahler-inspired orchestral storm piece.

"Lots of notes!" — everything microtonally detuned ±5-50 cents.
Massive forces: tremolo strings, hail pellets, thunder, wind,
brass fanfares, cowbells, timpani. Dense, overlapping, chaotic
but with an emotional melody trying to survive the storm.

Synthesis concepts:
  - Microtonal detuning: every note randomly ±5-50 cents off ET
  - Tremolo strings: rapid repeated tones with bow-pressure variation
  - Hail pellets: ultra-short bright staccato clusters
  - Thunder: low sub-bass rumble with envelope shaping
  - Wind: swept bandpass noise with slow modulation
  - Brass fanfare: bright partials with strong attack, wide vibrato
  - Cowbells: inharmonic metallic partials (Mahler 6th!)
  - Timpani: tuned drum with membrane modes

Structure (2:40 = 160s):
  0-15s    Pastoral calm — string melody, distant cowbells
  15-35s   Storm approaches — wind builds, first hail, timpani rumble
  35-70s   Full storm — tutti, hail clusters, thunder, tremolo, brass
  70-85s   Eye of storm — sudden quiet, cowbells, melody fragment
  85-130s  Second wave — even denser, full chaos, brass screaming
  130-150s Storm passes — thinning, melody returns bruised
  150-160s Aftermath — distant cowbells, one last chord

No samples, no DAW. Pure numpy/scipy. ~500+ events.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import argparse

SAMPLE_RATE = 44100
DURATION = 160  # 2:40

BASE_FREQ = 261.63  # C4

# ============================================================
# MICROTONAL SYSTEM
# ============================================================

def freq_from_pc_micro(pc, octave, rng, detune_range=30):
    """Frequency with microtonal detuning. detune_range in cents."""
    base = BASE_FREQ * (2 ** ((pc - 0) / 12.0 + (octave - 4)))
    detune_cents = rng.uniform(-detune_range, detune_range)
    return base * (2 ** (detune_cents / 1200.0)), detune_cents


# ============================================================
# TIMBRES
# ============================================================

# --- Strings (violin/viola/cello sections) ---
STRING_PROFILES = {
    "violin": {
        "harmonics": [1.0, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08, 0.05],
        "attack": 0.04,
        "noise_type": "bow",
        "noise_amount": 0.06,
        "vibrato_rate": 5.5,
        "vibrato_depth": 12,  # cents
        "vibrato_delay": 0.15,
    },
    "viola": {
        "harmonics": [1.0, 0.8, 0.55, 0.4, 0.25, 0.15, 0.08],
        "attack": 0.05,
        "noise_type": "bow",
        "noise_amount": 0.07,
        "vibrato_rate": 5.0,
        "vibrato_depth": 14,
        "vibrato_delay": 0.18,
    },
    "cello": {
        "harmonics": [1.0, 0.85, 0.6, 0.45, 0.3, 0.18, 0.1, 0.06],
        "attack": 0.06,
        "noise_type": "bow",
        "noise_amount": 0.08,
        "vibrato_rate": 4.5,
        "vibrato_depth": 16,
        "vibrato_delay": 0.2,
    },
    "bass": {
        "harmonics": [1.0, 0.9, 0.65, 0.5, 0.35, 0.2, 0.12],
        "attack": 0.08,
        "noise_type": "bow",
        "noise_amount": 0.09,
        "vibrato_rate": 4.0,
        "vibrato_depth": 10,
        "vibrato_delay": 0.25,
    },
}

# --- Brass ---
BRASS_PROFILES = {
    "horn": {
        "harmonics": [1.0, 0.9, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08, 0.05],
        "attack": 0.03,
        "noise_type": "breath",
        "noise_amount": 0.04,
        "vibrato_rate": 4.5,
        "vibrato_depth": 8,
        "vibrato_delay": 0.3,
        "brightness_boost": 1.0,
    },
    "trumpet": {
        "harmonics": [1.0, 0.95, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12, 0.07],
        "attack": 0.015,
        "noise_type": "breath",
        "noise_amount": 0.05,
        "vibrato_rate": 5.0,
        "vibrato_depth": 6,
        "vibrato_delay": 0.2,
        "brightness_boost": 1.3,
    },
    "trombone": {
        "harmonics": [1.0, 0.85, 0.7, 0.6, 0.45, 0.3, 0.2, 0.12, 0.07],
        "attack": 0.025,
        "noise_type": "breath",
        "noise_amount": 0.06,
        "vibrato_rate": 4.0,
        "vibrato_depth": 10,
        "vibrato_delay": 0.25,
        "brightness_boost": 1.1,
    },
}


# ============================================================
# SYNTHESIS FUNCTIONS
# ============================================================

def make_string_tone(t, start, freq, duration, amplitude, profile_name, rng,
                     tremolo=False, tremolo_speed=12.0):
    """Synthesize a string tone with optional tremolo."""
    sr = SAMPLE_RATE
    profile = STRING_PROFILES[profile_name]
    s = int(start * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr
    voice = np.zeros(len(t))

    # Pitch drift (organic)
    drift = np.cumsum(rng.randn(n) * 0.3) * 0.5
    drift = np.clip(drift, -15, 15)  # ±15 cents max
    freq_curve = freq * (2 ** (drift / 1200))

    # Vibrato (delayed onset)
    vib_rate = profile["vibrato_rate"] + rng.uniform(-0.3, 0.3)
    vib_depth = profile["vibrato_depth"]
    vib_delay = profile["vibrato_delay"]
    vib_onset = np.clip((local_t - vib_delay) / 0.3, 0, 1)
    # Noise-modulated vibrato rate
    vib_noise = rng.randn(n) * 0.2
    vib_phase = np.cumsum(2 * np.pi * (vib_rate + vib_noise) / sr)
    vibrato = vib_onset * vib_depth * np.sin(vib_phase)
    freq_curve *= (2 ** (vibrato / 1200))

    # Per-partial synthesis
    for i, h_amp in enumerate(profile["harmonics"]):
        partial_num = i + 1
        # Inharmonicity
        inharmonicity = 0.0003
        partial_freq = freq_curve * partial_num * np.sqrt(1 + inharmonicity * partial_num**2)
        phase = np.cumsum(2 * np.pi * partial_freq / sr)

        # Upper partials decay faster
        partial_decay = np.exp(-local_t * 0.3 * partial_num)
        partial_audio = h_amp * partial_decay * np.sin(phase)
        voice[s:end] += partial_audio[:n]

    # Attack transient — bow noise
    atk_dur = int(profile["attack"] * sr)
    if atk_dur > 0 and atk_dur <= n:
        noise = rng.randn(atk_dur) * profile["noise_amount"]
        noise *= np.exp(-np.linspace(0, 6, atk_dur))
        # Bandpass around 2-4kHz for bow scratchiness
        nyq = sr / 2
        lo = 1500 / nyq
        hi = min(4000 / nyq, 0.95)
        if lo < hi:
            b, a = butter(2, [lo, hi], btype='band')
            noise = lfilter(b, a, noise)
        voice[s:s + atk_dur] += noise[:min(atk_dur, n)] * amplitude

    # Amplitude envelope
    atk_samp = int(profile["attack"] * sr)
    env = np.ones(n)
    if atk_samp > 0 and atk_samp < n:
        env[:atk_samp] = np.linspace(0, 1, atk_samp)
    # Sustained shape with gentle release
    rel_samp = min(int(0.08 * sr), n // 4)
    if rel_samp > 0:
        env[-rel_samp:] *= np.linspace(1, 0, rel_samp)

    voice[s:end] *= env * amplitude

    # Tremolo — rapid bow-stroke repetition
    if tremolo:
        trem_t = local_t
        # Slightly irregular tremolo speed
        trem_noise = rng.randn(n) * 0.5
        trem_phase = np.cumsum(2 * np.pi * (tremolo_speed + trem_noise) / sr)
        trem_env = 0.55 + 0.45 * np.abs(np.sin(trem_phase))
        voice[s:end] *= trem_env

    return voice


def make_brass_tone(t, start, freq, duration, amplitude, profile_name, rng):
    """Synthesize a brass tone — bright, powerful."""
    sr = SAMPLE_RATE
    profile = BRASS_PROFILES[profile_name]
    s = int(start * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr
    voice = np.zeros(len(t))

    # Pitch drift
    drift = np.cumsum(rng.randn(n) * 0.2) * 0.4
    drift = np.clip(drift, -10, 10)
    freq_curve = freq * (2 ** (drift / 1200))

    # Vibrato
    vib_rate = profile["vibrato_rate"]
    vib_depth = profile["vibrato_depth"]
    vib_delay = profile["vibrato_delay"]
    vib_onset = np.clip((local_t - vib_delay) / 0.3, 0, 1)
    vibrato = vib_onset * vib_depth * np.sin(2 * np.pi * vib_rate * local_t)
    freq_curve *= (2 ** (vibrato / 1200))

    # Brass has strong upper harmonics
    brightness = profile.get("brightness_boost", 1.0)
    for i, h_amp in enumerate(profile["harmonics"]):
        partial_num = i + 1
        partial_freq = freq_curve * partial_num
        phase = np.cumsum(2 * np.pi * partial_freq / sr)
        # Brass: upper partials actually strengthen during attack
        partial_env = np.ones(n)
        if partial_num > 3:
            # Higher partials bloom with amplitude
            bloom_time = int(0.1 * sr)
            if bloom_time < n:
                partial_env[:bloom_time] = np.linspace(0.3, 1.0, bloom_time)
        voice[s:end] += h_amp * brightness * partial_env * np.sin(phase)

    # Breath attack transient
    atk_dur = int(profile["attack"] * sr)
    if atk_dur > 0 and atk_dur <= n:
        noise = rng.randn(atk_dur) * profile["noise_amount"]
        noise *= np.exp(-np.linspace(0, 8, atk_dur))
        voice[s:s + atk_dur] += noise[:min(atk_dur, n)] * amplitude

    # Envelope — brass has a sharp attack, sustained body, quick release
    env = np.ones(n)
    atk = int(profile["attack"] * sr)
    if atk > 0 and atk < n:
        env[:atk] = np.linspace(0, 1, atk) ** 0.5  # convex attack
    rel = min(int(0.05 * sr), n // 4)
    if rel > 0:
        env[-rel:] *= np.linspace(1, 0, rel)

    voice[s:end] *= env * amplitude

    return voice


def make_hail_pellet(t, time_pos, freq, amplitude, rng):
    """Single hail pellet — ultra-short, bright, percussive."""
    sr = SAMPLE_RATE
    s = int(time_pos * sr)
    dur = int(rng.uniform(0.008, 0.025) * sr)  # 8-25ms
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    voice = np.zeros(len(t))

    # Bright click + pitched ring
    env = np.exp(-local_t * rng.uniform(60, 150))

    # 2-3 inharmonic partials
    n_partials = rng.randint(2, 4)
    for _ in range(n_partials):
        p_freq = freq * rng.uniform(0.8, 3.0)
        p_amp = rng.uniform(0.3, 1.0)
        voice[s:end] += p_amp * env * np.sin(2 * np.pi * p_freq * local_t)

    # Noise click
    noise = rng.randn(n) * np.exp(-local_t * 200)
    voice[s:end] += noise * 0.3

    voice[s:end] *= amplitude
    return voice


def make_thunder(t, start, duration, amplitude, rng):
    """Low sub-bass rumble with irregular envelope."""
    sr = SAMPLE_RATE
    s = int(start * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    voice = np.zeros(len(t))

    # Sub-bass rumble: multiple low frequencies
    for _ in range(4):
        freq = rng.uniform(25, 60)
        amp = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        voice[s:end] += amp * np.sin(2 * np.pi * freq * local_t + phase)

    # Irregular amplitude modulation — thunder crackles
    mod_freq = rng.uniform(3, 12)
    mod = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * mod_freq * local_t +
                                      rng.randn(n) * 0.5))
    voice[s:end] *= mod

    # Envelope — sharp attack, long decay
    env = np.exp(-local_t * (1.0 / duration) * 3)
    atk = min(int(0.02 * sr), n)
    if atk > 0:
        env[:atk] *= np.linspace(0, 1, atk)
    voice[s:end] *= env * amplitude

    # Low-pass to keep it sub-bass
    nyq = sr / 2
    b_lp, a_lp = butter(3, 120 / nyq, btype='low')
    rumble = np.zeros_like(t)
    rumble[s:end] = voice[s:end]
    rumble = lfilter(b_lp, a_lp, rumble)

    return rumble


def make_wind(t, start, duration, amplitude, rng, intensity=0.5):
    """Swept bandpass noise — storm wind."""
    sr = SAMPLE_RATE
    s = int(start * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    # Generate broadband noise
    noise = rng.randn(n)

    # Slow sweep of center frequency
    sweep_center = 400 + 800 * intensity
    sweep_range = 300 * intensity
    sweep_rate = rng.uniform(0.1, 0.4)
    center = sweep_center + sweep_range * np.sin(2 * np.pi * sweep_rate * local_t)

    # Filter in chunks
    chunk_size = int(0.02 * sr)
    wind = np.zeros(n)
    nyq = sr / 2

    for i in range(0, n, chunk_size):
        ce = min(i + chunk_size, n)
        cn = ce - i
        freq = float(np.mean(center[i:ce]))
        bw = freq * 0.5
        lo = max((freq - bw), 50) / nyq
        hi = min((freq + bw), nyq * 0.9) / nyq
        if lo < hi and lo > 0 and cn > 12:
            b, a = butter(2, [lo, hi], btype='band')
            wind[i:ce] = lfilter(b, a, noise[i:ce])
        else:
            wind[i:ce] = noise[i:ce] * 0.05

    # Gusting amplitude modulation
    gust_rate = rng.uniform(0.3, 1.5)
    gust = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * gust_rate * local_t +
                                               rng.randn(n) * 0.3))
    wind *= gust

    # Envelope
    env = np.ones(n)
    fade_in = min(int(1.0 * sr), n // 3)
    fade_out = min(int(1.5 * sr), n // 3)
    if fade_in > 0:
        env[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out > 0:
        env[-fade_out:] *= np.linspace(1, 0, fade_out)

    out = np.zeros_like(t)
    out[s:end] = wind * env * amplitude
    return out


def make_cowbell(t, time_pos, freq, duration, amplitude, rng):
    """Mahler cowbell — inharmonic metallic partials."""
    sr = SAMPLE_RATE
    s = int(time_pos * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    voice = np.zeros(len(t))

    # Cowbell partials — distinctly inharmonic
    partials = [
        (1.0,   1.0,  3.0),
        (1.504, 0.7,  4.0),
        (1.836, 0.5,  3.5),
        (2.0,   0.3,  5.0),
        (2.536, 0.4,  4.5),
        (3.17,  0.2,  6.0),
    ]

    for ratio, amp, decay in partials:
        p_freq = freq * ratio * (1 + rng.uniform(-0.003, 0.003))
        env = np.exp(-decay * local_t)
        voice[s:end] += amp * env * np.sin(2 * np.pi * p_freq * local_t)

    # Strike transient
    strike_dur = int(0.005 * sr)
    if strike_dur > 0 and strike_dur <= n:
        strike = rng.randn(strike_dur) * np.exp(-np.linspace(0, 15, strike_dur))
        voice[s:s + strike_dur] += strike * 0.15 * amplitude

    voice[s:end] *= amplitude

    # Anti-click fadeout
    fade = min(int(0.01 * sr), n)
    if fade > 0:
        voice[end - fade:end] *= np.linspace(1, 0, fade)

    return voice


def make_timpani(t, time_pos, freq, duration, amplitude, rng):
    """Timpani — tuned drum with membrane modes."""
    sr = SAMPLE_RATE
    s = int(time_pos * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    voice = np.zeros(len(t))

    # Membrane modes — inharmonic ratios for circular membrane
    modes = [
        (1.0,   1.0,  2.5),   # fundamental
        (1.504, 0.6,  3.5),   # (1,1) mode
        (1.742, 0.3,  4.0),   # (2,1) mode
        (2.0,   0.5,  3.0),   # (0,2) mode
        (2.296, 0.2,  5.0),   # (3,1) mode
    ]

    for ratio, amp, decay in modes:
        mode_freq = freq * ratio
        env = np.exp(-decay * local_t)
        voice[s:end] += amp * env * np.sin(2 * np.pi * mode_freq * local_t)

    # Stick impact
    impact_dur = int(0.008 * sr)
    if impact_dur > 0 and impact_dur <= n:
        impact = rng.randn(impact_dur)
        impact *= np.exp(-np.linspace(0, 12, impact_dur))
        nyq = sr / 2
        b, a = butter(2, min(3000 / nyq, 0.9), btype='low')
        impact = lfilter(b, a, impact)
        voice[s:s + impact_dur] += impact * 0.2 * amplitude

    voice[s:end] *= amplitude

    # Fade
    fade = min(int(0.02 * sr), n)
    if fade > 0:
        voice[end - fade:end] *= np.linspace(1, 0, fade)

    return voice


# ============================================================
# REVERB — large concert hall
# ============================================================

def concert_hall_reverb(audio, decay=0.45, sample_rate=44100):
    """Large hall reverb — long delays, rich diffusion."""
    delays_ms = [29, 43, 61, 83, 109, 139, 179, 227, 283, 353, 431]
    wet = np.zeros_like(audio)

    for i, d in enumerate(delays_ms):
        delay_samp = int(d * sample_rate / 1000)
        gain = decay * (0.85 ** i)
        delayed = np.zeros_like(audio)
        if delay_samp < len(audio):
            delayed[delay_samp:] = audio[:-delay_samp] * gain
        wet += delayed

    return audio * 0.78 + wet * 0.22


# ============================================================
# SCALES
# ============================================================

# D minor — Mahler's tragic key
D_MINOR = [2, 4, 5, 7, 9, 10, 0]   # D E F G A Bb C
# Bb major — storm resolution key
Bb_MAJOR = [10, 0, 2, 3, 5, 7, 9]  # Bb C D Eb F G A
# D major — Mahler's heroic resolution
D_MAJOR = [2, 4, 6, 7, 9, 11, 1]   # D E F# G A B C#

# Mahler-style melody — long, yearning, in D minor
# (scale_degree, octave_offset, beats, accent)
MAHLER_MELODY = [
    (0, 0, 2.0, 0.7),   # D
    (2, 0, 1.0, 0.5),   # F
    (4, 0, 1.5, 0.6),   # A
    (3, 0, 0.5, 0.4),   # G
    (2, 0, 3.0, 0.8),   # F — yearning hold
    (1, 0, 1.0, 0.5),   # E
    (0, 0, 1.0, 0.6),   # D
    (5, -1, 1.0, 0.4),  # Bb (below)
    (0, 0, 3.0, 0.7),   # D — resolve
    # Second phrase — reaching higher
    (4, 0, 1.5, 0.7),   # A
    (5, 0, 1.0, 0.6),   # Bb
    (6, 0, 0.5, 0.5),   # C
    (0, 1, 2.0, 0.9),   # D (high) — climax
    (6, 0, 1.0, 0.5),   # C
    (4, 0, 1.0, 0.6),   # A
    (2, 0, 1.5, 0.5),   # F
    (0, 0, 3.0, 0.7),   # D — home
]


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_piece(seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, DURATION, DURATION * SAMPLE_RATE, endpoint=False)
    audio = np.zeros_like(t)
    events = []

    # ================================================================
    # SECTION 1: PASTORAL CALM (0-15s)
    # Gentle string melody + distant cowbells
    # ================================================================

    # Melody on violin, with viola doubling a 3rd below
    melody_time = 1.0
    bpm = 52  # Slow, expansive
    beat_dur = 60.0 / bpm

    for deg, oct_off, dur_beats, accent in MAHLER_MELODY[:9]:
        if melody_time > 14:
            break
        pc = D_MINOR[deg % len(D_MINOR)]
        octave = 5 + oct_off
        freq, detune = freq_from_pc_micro(pc, octave, rng, detune_range=15)
        dur = dur_beats * beat_dur
        amp = accent * 0.16

        audio += make_string_tone(t, melody_time, freq, dur, amp, "violin", rng)
        events.append({'time': melody_time, 'pc': pc, 'octave': octave,
                        'duration': dur, 'amplitude': amp, 'type': 'violin',
                        'category': 'melody'})

        # Viola — a 3rd below
        via_deg = (deg - 2) % len(D_MINOR)
        via_pc = D_MINOR[via_deg]
        via_oct = octave - 1 if via_pc > pc else octave
        via_freq, _ = freq_from_pc_micro(via_pc, via_oct, rng, detune_range=20)
        audio += make_string_tone(t, melody_time + 0.05, via_freq, dur * 0.9,
                                   amp * 0.6, "viola", rng)
        events.append({'time': melody_time + 0.05, 'pc': via_pc, 'octave': via_oct,
                        'duration': dur * 0.9, 'amplitude': amp * 0.6, 'type': 'viola',
                        'category': 'accompaniment'})

        melody_time += dur

    # Distant cowbells (Mahler 6th!)
    for _ in range(6):
        cb_time = rng.uniform(2, 14)
        cb_freq, _ = freq_from_pc_micro(rng.choice([7, 9, 0, 2]), 5, rng, 25)
        cb_dur = rng.uniform(0.8, 2.0)
        cb_amp = rng.uniform(0.02, 0.04)
        audio += make_cowbell(t, cb_time, cb_freq, cb_dur, cb_amp, rng)
        events.append({'time': cb_time, 'pc': 7, 'octave': 5,
                        'duration': cb_dur, 'amplitude': cb_amp, 'type': 'cowbell',
                        'category': 'color'})

    # Cello pedal tone on D
    d_freq, _ = freq_from_pc_micro(2, 3, rng, 10)
    audio += make_string_tone(t, 0.5, d_freq, 14, 0.08, "cello", rng)
    events.append({'time': 0.5, 'pc': 2, 'octave': 3,
                    'duration': 14, 'amplitude': 0.08, 'type': 'cello',
                    'category': 'pedal'})

    # ================================================================
    # SECTION 2: STORM APPROACHES (15-35s)
    # Wind builds, first hail, timpani rumble
    # ================================================================

    # Wind noise building
    audio += make_wind(t, 14, 22, 0.06, rng, intensity=0.3)
    audio += make_wind(t, 20, 18, 0.10, rng, intensity=0.5)

    # Timpani rumbles
    for _ in range(5):
        timp_time = rng.uniform(16, 34)
        timp_freq, _ = freq_from_pc_micro(2, 2, rng, 15)  # Low D
        timp_dur = rng.uniform(1.0, 3.0)
        timp_amp = rng.uniform(0.06, 0.12)
        audio += make_timpani(t, timp_time, timp_freq, timp_dur, timp_amp, rng)
        events.append({'time': timp_time, 'pc': 2, 'octave': 2,
                        'duration': timp_dur, 'amplitude': timp_amp, 'type': 'timpani',
                        'category': 'percussion'})

    # First hail pellets — scattered
    for _ in range(30):
        h_time = rng.uniform(22, 35)
        h_freq = rng.uniform(2000, 8000)
        h_amp = rng.uniform(0.01, 0.04)
        audio += make_hail_pellet(t, h_time, h_freq, h_amp, rng)
        events.append({'time': h_time, 'pc': rng.randint(0, 12), 'octave': 6,
                        'duration': 0.02, 'amplitude': h_amp, 'type': 'hail',
                        'category': 'storm'})

    # Tremolo strings building tension
    for _ in range(8):
        trem_time = rng.uniform(18, 34)
        trem_pc = rng.choice(D_MINOR)
        trem_oct = rng.choice([3, 4])
        trem_freq, _ = freq_from_pc_micro(trem_pc, trem_oct, rng, 30)
        trem_dur = rng.uniform(2.0, 5.0)
        trem_amp = rng.uniform(0.04, 0.09)
        prof = rng.choice(["violin", "viola", "cello"])
        audio += make_string_tone(t, trem_time, trem_freq, trem_dur, trem_amp,
                                   prof, rng, tremolo=True, tremolo_speed=14)
        events.append({'time': trem_time, 'pc': trem_pc, 'octave': trem_oct,
                        'duration': trem_dur, 'amplitude': trem_amp,
                        'type': prof, 'category': 'tremolo'})

    # ================================================================
    # SECTION 3: FULL STORM (35-70s)
    # Tutti — hail clusters, thunder, tremolo, brass
    # ================================================================

    # Dense hail
    for _ in range(120):
        h_time = rng.uniform(35, 70)
        h_freq = rng.uniform(1500, 10000)
        h_amp = rng.uniform(0.02, 0.06)
        audio += make_hail_pellet(t, h_time, h_freq, h_amp, rng)
        events.append({'time': h_time, 'pc': rng.randint(0, 12), 'octave': 7,
                        'duration': 0.02, 'amplitude': h_amp, 'type': 'hail',
                        'category': 'storm'})

    # Thunder
    thunder_times = [36, 42, 48, 53, 60, 66]
    for tt in thunder_times:
        dur = rng.uniform(3, 6)
        amp = rng.uniform(0.08, 0.15)
        audio += make_thunder(t, tt, dur, amp, rng)
        events.append({'time': tt, 'pc': 2, 'octave': 1,
                        'duration': dur, 'amplitude': amp, 'type': 'thunder',
                        'category': 'storm'})

    # Full wind
    audio += make_wind(t, 34, 38, 0.14, rng, intensity=0.8)

    # Massive tremolo strings — full section
    for _ in range(25):
        trem_time = rng.uniform(35, 70)
        trem_pc = rng.choice(D_MINOR)
        trem_oct = rng.choice([3, 4, 5])
        trem_freq, _ = freq_from_pc_micro(trem_pc, trem_oct, rng, 40)
        trem_dur = rng.uniform(1.5, 4.0)
        trem_amp = rng.uniform(0.05, 0.12)
        prof = rng.choice(["violin", "viola", "cello", "bass"])
        trem_spd = rng.uniform(10, 18)
        audio += make_string_tone(t, trem_time, trem_freq, trem_dur, trem_amp,
                                   prof, rng, tremolo=True, tremolo_speed=trem_spd)
        events.append({'time': trem_time, 'pc': trem_pc, 'octave': trem_oct,
                        'duration': trem_dur, 'amplitude': trem_amp,
                        'type': prof, 'category': 'tremolo'})

    # Brass fanfares cutting through the storm
    brass_entries = [
        (38, "horn", 2, 4, 3.0, 0.13),    # D4 horn call
        (42, "trumpet", 9, 5, 2.0, 0.15),  # A5 trumpet
        (47, "trombone", 2, 3, 4.0, 0.12), # D3 trombone
        (52, "trumpet", 5, 5, 1.5, 0.16),  # F5 trumpet cry
        (55, "horn", 10, 4, 3.0, 0.14),    # Bb4 horn
        (58, "trumpet", 0, 6, 2.5, 0.17),  # C6 trumpet scream
        (62, "trombone", 7, 3, 3.5, 0.13), # G3 trombone
        (65, "horn", 2, 5, 2.0, 0.15),     # D5 horn
    ]
    for bt, prof, pc, oct, dur, amp in brass_entries:
        freq, _ = freq_from_pc_micro(pc, oct, rng, 25)
        audio += make_brass_tone(t, bt, freq, dur, amp, prof, rng)
        events.append({'time': bt, 'pc': pc, 'octave': oct,
                        'duration': dur, 'amplitude': amp,
                        'type': prof, 'category': 'brass'})

    # Timpani — marcato
    for _ in range(10):
        timp_time = rng.uniform(35, 70)
        timp_pc = rng.choice([2, 9, 7])  # D, A, G
        timp_freq, _ = freq_from_pc_micro(timp_pc, 2, rng, 20)
        audio += make_timpani(t, timp_time, timp_freq, rng.uniform(1, 3),
                               rng.uniform(0.10, 0.18), rng)
        events.append({'time': timp_time, 'pc': timp_pc, 'octave': 2,
                        'duration': 2.0, 'amplitude': 0.14, 'type': 'timpani',
                        'category': 'percussion'})

    # ================================================================
    # SECTION 4: EYE OF STORM (70-85s)
    # Sudden quiet, cowbells, melody fragment
    # ================================================================

    # Cowbells — closer now, Mahler's Alpine memory
    for _ in range(10):
        cb_time = rng.uniform(72, 84)
        cb_pc = rng.choice([7, 0, 4, 9])
        cb_freq, _ = freq_from_pc_micro(cb_pc, 5, rng, 20)
        cb_amp = rng.uniform(0.03, 0.06)
        audio += make_cowbell(t, cb_time, cb_freq, rng.uniform(1, 3), cb_amp, rng)
        events.append({'time': cb_time, 'pc': cb_pc, 'octave': 5,
                        'duration': 1.5, 'amplitude': cb_amp, 'type': 'cowbell',
                        'category': 'color'})

    # Melody fragment on horn — broken, searching
    horn_melody_time = 73.0
    for deg, oct_off, dur_beats, accent in MAHLER_MELODY[9:13]:
        if horn_melody_time > 83:
            break
        pc = D_MINOR[deg % len(D_MINOR)]
        octave = 4 + oct_off
        freq, _ = freq_from_pc_micro(pc, octave, rng, 20)
        dur = dur_beats * beat_dur * 1.3  # Slower, more searching
        amp = accent * 0.10

        audio += make_brass_tone(t, horn_melody_time, freq, dur, amp, "horn", rng)
        events.append({'time': horn_melody_time, 'pc': pc, 'octave': octave,
                        'duration': dur, 'amplitude': amp, 'type': 'horn',
                        'category': 'melody'})
        horn_melody_time += dur

    # Quiet wind — still present
    audio += make_wind(t, 70, 16, 0.04, rng, intensity=0.2)

    # ================================================================
    # SECTION 5: SECOND WAVE (85-130s)
    # Even denser, full chaos, brass screaming
    # ================================================================

    # Massive hail
    for _ in range(180):
        h_time = rng.uniform(85, 130)
        h_freq = rng.uniform(1000, 12000)
        h_amp = rng.uniform(0.02, 0.07)
        audio += make_hail_pellet(t, h_time, h_freq, h_amp, rng)
        events.append({'time': h_time, 'pc': rng.randint(0, 12), 'octave': 7,
                        'duration': 0.02, 'amplitude': h_amp, 'type': 'hail',
                        'category': 'storm'})

    # Thunder — bigger
    for tt in [86, 90, 95, 100, 106, 112, 118, 125]:
        dur = rng.uniform(4, 8)
        amp = rng.uniform(0.10, 0.20)
        audio += make_thunder(t, tt, dur, amp, rng)
        events.append({'time': tt, 'pc': 2, 'octave': 1,
                        'duration': dur, 'amplitude': amp, 'type': 'thunder',
                        'category': 'storm'})

    # Howling wind
    audio += make_wind(t, 84, 48, 0.18, rng, intensity=1.0)
    audio += make_wind(t, 90, 42, 0.12, rng, intensity=0.7)

    # Tremolo strings — doubled sections, massive
    for _ in range(35):
        trem_time = rng.uniform(85, 130)
        trem_pc = rng.choice(D_MINOR)
        trem_oct = rng.choice([3, 4, 5])
        trem_freq, _ = freq_from_pc_micro(trem_pc, trem_oct, rng, 50)  # WIDE microtones
        trem_dur = rng.uniform(1.0, 4.0)
        trem_amp = rng.uniform(0.06, 0.14)
        prof = rng.choice(["violin", "viola", "cello", "bass"])
        audio += make_string_tone(t, trem_time, trem_freq, trem_dur, trem_amp,
                                   prof, rng, tremolo=True,
                                   tremolo_speed=rng.uniform(12, 20))
        events.append({'time': trem_time, 'pc': trem_pc, 'octave': trem_oct,
                        'duration': trem_dur, 'amplitude': trem_amp,
                        'type': prof, 'category': 'tremolo'})

    # Brass — more, louder, higher
    brass_storm2 = [
        (87, "trumpet", 2, 6, 2.0, 0.18),   # D6
        (90, "horn", 9, 5, 3.0, 0.16),       # A5
        (93, "trombone", 5, 3, 4.0, 0.15),   # F3
        (96, "trumpet", 7, 6, 1.5, 0.20),    # G6 — screaming
        (99, "horn", 2, 5, 2.5, 0.17),       # D5
        (102, "trumpet", 0, 6, 2.0, 0.19),   # C6
        (105, "trombone", 10, 3, 3.0, 0.14), # Bb3
        (108, "trumpet", 5, 6, 1.0, 0.21),   # F6 — highest scream
        (110, "horn", 7, 4, 4.0, 0.16),      # G4
        (114, "trumpet", 2, 6, 3.0, 0.20),   # D6
        (118, "trombone", 9, 3, 3.5, 0.15),  # A3
        (122, "horn", 5, 5, 2.0, 0.18),      # F5
        (125, "trumpet", 4, 5, 2.0, 0.16),   # E5 — Picardy turn?
    ]
    for bt, prof, pc, oct, dur, amp in brass_storm2:
        freq, _ = freq_from_pc_micro(pc, oct, rng, 35)
        audio += make_brass_tone(t, bt, freq, dur, amp, prof, rng)
        events.append({'time': bt, 'pc': pc, 'octave': oct,
                        'duration': dur, 'amplitude': amp,
                        'type': prof, 'category': 'brass'})

    # Timpani rolls
    for _ in range(15):
        timp_time = rng.uniform(85, 128)
        timp_pc = rng.choice([2, 9, 5])
        timp_freq, _ = freq_from_pc_micro(timp_pc, 2, rng, 25)
        audio += make_timpani(t, timp_time, timp_freq, rng.uniform(1, 3),
                               rng.uniform(0.12, 0.22), rng)
        events.append({'time': timp_time, 'pc': timp_pc, 'octave': 2,
                        'duration': 2.0, 'amplitude': 0.17, 'type': 'timpani',
                        'category': 'percussion'})

    # ================================================================
    # SECTION 6: STORM PASSES (130-150s)
    # Thinning, melody returns bruised, shifting to D major
    # ================================================================

    # Fading wind
    audio += make_wind(t, 128, 20, 0.08, rng, intensity=0.4)

    # Scattered last hail
    for _ in range(15):
        h_time = rng.uniform(130, 145)
        h_freq = rng.uniform(2000, 6000)
        h_amp = rng.uniform(0.01, 0.03)
        audio += make_hail_pellet(t, h_time, h_freq, h_amp, rng)
        events.append({'time': h_time, 'pc': rng.randint(0, 12), 'octave': 6,
                        'duration': 0.02, 'amplitude': h_amp, 'type': 'hail',
                        'category': 'storm'})

    # Melody returns — on cello, bruised, now shifting toward D major
    mel_time = 132.0
    scale = D_MAJOR  # The Mahler turn to major
    for deg, oct_off, dur_beats, accent in MAHLER_MELODY:
        if mel_time > 149:
            break
        pc = scale[deg % len(scale)]
        octave = 4 + oct_off
        freq, _ = freq_from_pc_micro(pc, octave, rng, 20)
        dur = dur_beats * beat_dur * 1.5  # Slower
        amp = accent * 0.12

        audio += make_string_tone(t, mel_time, freq, dur, amp, "cello", rng)
        events.append({'time': mel_time, 'pc': pc, 'octave': octave,
                        'duration': dur, 'amplitude': amp, 'type': 'cello',
                        'category': 'melody'})

        # Violin doubling — an octave up, very quiet
        vin_freq, _ = freq_from_pc_micro(pc, octave + 1, rng, 25)
        audio += make_string_tone(t, mel_time + 0.03, vin_freq, dur * 0.8,
                                   amp * 0.4, "violin", rng)
        events.append({'time': mel_time + 0.03, 'pc': pc, 'octave': octave + 1,
                        'duration': dur * 0.8, 'amplitude': amp * 0.4, 'type': 'violin',
                        'category': 'accompaniment'})

        mel_time += dur

    # One last distant timpani
    audio += make_timpani(t, 140, freq_from_pc_micro(2, 2, rng, 10)[0],
                           4.0, 0.06, rng)

    # ================================================================
    # SECTION 7: AFTERMATH (150-160s)
    # Distant cowbells, one last D major chord
    # ================================================================

    # Distant cowbells
    for _ in range(5):
        cb_time = rng.uniform(150, 158)
        cb_freq, _ = freq_from_pc_micro(rng.choice([2, 6, 9]), 5, rng, 15)
        audio += make_cowbell(t, cb_time, cb_freq, rng.uniform(1, 2.5),
                               rng.uniform(0.02, 0.04), rng)
        events.append({'time': cb_time, 'pc': 2, 'octave': 5,
                        'duration': 1.5, 'amplitude': 0.03, 'type': 'cowbell',
                        'category': 'color'})

    # Final D major chord — strings, very quiet, resolving
    chord_time = 153.0
    chord_pcs = [2, 6, 9]  # D F# A
    for cpc in chord_pcs:
        for oct in [3, 4, 5]:
            freq, _ = freq_from_pc_micro(cpc, oct, rng, 12)
            amp = 0.05 if oct == 4 else 0.03
            prof = "cello" if oct == 3 else ("viola" if oct == 4 else "violin")
            audio += make_string_tone(t, chord_time, freq, 7.0, amp, prof, rng)
            events.append({'time': chord_time, 'pc': cpc, 'octave': oct,
                            'duration': 7.0, 'amplitude': amp, 'type': prof,
                            'category': 'chord'})

    # ============================================================
    # MIX AND MASTER
    # ============================================================

    # Concert hall reverb
    audio = concert_hall_reverb(audio, decay=0.45, sample_rate=SAMPLE_RATE)

    # Normalize — this one should be LOUD but controlled
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    # Gentle warmth
    nyq = SAMPLE_RATE / 2
    b, a = butter(1, 14000 / nyq, btype='low')
    audio = lfilter(b, a, audio)

    return audio, events


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mahler Hailstorm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Generating 'Hagelstürme'...")
    print(f"  Seed: {args.seed}")
    print(f"  Duration: {DURATION}s")

    audio, events = generate_piece(seed=args.seed)

    output = args.output or "mahler_hail_01.wav"
    wavfile.write(output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"  Events: {len(events)}")
    print(f"  Output: {output}")
    print("Done.")
