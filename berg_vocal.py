"""
berg_vocal.py — Berg-inspired composition with vocal synthesis

Adds formant-based voice simulation to the synthesis engine:
  - Glottal pulse source (not sine waves — rich harmonic spectrum)
  - Formant resonance filtering (vowel shapes: ah, ee, oh, oo, mm)
  - Vowel morphing over note lifetime
  - Breathiness (aspiration noise mixed with source)
  - Cycle-to-cycle jitter (micro-randomness in vocal fold timing)
  - Body resonance filtering for instruments too

Four voice types: soprano, alto, tenor, bass
Combined with upgraded instruments for a Berg choral-orchestral piece.

Usage:
    python berg_vocal.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, sosfilt, butter
import random
import argparse

from webern_pointillism import (
    TIMBRES as INST_TIMBRES, SAMPLE_RATE, simple_reverb,
    freq_from_pitch_class, generate_noise
)

DURATION = 100  # seconds


# =====================================================================
# FORMANT DATA — vowel resonant frequencies
# Each vowel: list of (freq_hz, bandwidth_hz, amplitude)
# Based on Peterson & Barney (1952) + Fant (1960) measurements
# =====================================================================
VOWELS = {
    "ah": [  # as in "father" — open, warm
        (800,  80,  1.0),
        (1150, 90,  0.50),
        (2800, 120, 0.18),
        (3500, 130, 0.10),
        (4950, 140, 0.04),
    ],
    "ee": [  # as in "see" — bright, forward
        (270,  60,  1.0),
        (2300, 100, 0.35),
        (3000, 120, 0.15),
        (3700, 130, 0.08),
        (4950, 140, 0.03),
    ],
    "oh": [  # as in "go" — round, dark
        (500,  70,  1.0),
        (700,  80,  0.40),
        (2800, 100, 0.12),
        (3500, 130, 0.06),
        (4950, 140, 0.03),
    ],
    "oo": [  # as in "moon" — closed, deep
        (300,  50,  1.0),
        (600,  70,  0.30),
        (2300, 100, 0.08),
        (3500, 120, 0.04),
        (4950, 140, 0.02),
    ],
    "mm": [  # humming — nasal, closed
        (250,  60,  1.0),
        (1700, 120, 0.10),
        (2500, 130, 0.04),
        (3300, 140, 0.02),
    ],
    "eh": [  # as in "bed" — mid, neutral
        (530,  60,  1.0),
        (1850, 100, 0.35),
        (2500, 120, 0.12),
        (3500, 130, 0.06),
        (4950, 140, 0.03),
    ],
}

# Shift formants for different voice types (Hz offset)
VOICE_FORMANT_SHIFT = {
    "soprano": 1.15,   # formants shifted up ~15%
    "alto":    1.0,     # reference
    "tenor":   0.88,    # shifted down
    "bass":    0.78,    # shifted down more
}


# =====================================================================
# VOICE TIMBRES
# =====================================================================
VOICE_TIMBRES = [
    {
        "name": "soprano",
        "type": "voice",
        "octave_range": (4, 5),
        "vowel_sequence": ["mm", "ah", "ee", "ah", "mm"],
        "breathiness": 0.07,
        "jitter": 0.008,           # cycle-to-cycle pitch randomness
        "shimmer": 0.06,           # cycle-to-cycle amplitude randomness
        "vibrato_rate": 5.8,
        "vibrato_depth": 18,       # cents — operatic vibrato is wide
        "vibrato_delay": 0.15,
        "drift": 4,
        "attack": 0.08,
        "formant_shift": "soprano",
    },
    {
        "name": "alto",
        "type": "voice",
        "octave_range": (3, 4),
        "vowel_sequence": ["oh", "ah", "eh", "mm"],
        "breathiness": 0.06,
        "jitter": 0.010,
        "shimmer": 0.07,
        "vibrato_rate": 5.2,
        "vibrato_depth": 15,
        "vibrato_delay": 0.2,
        "drift": 5,
        "attack": 0.10,
        "formant_shift": "alto",
    },
    {
        "name": "tenor",
        "type": "voice",
        "octave_range": (3, 4),
        "vowel_sequence": ["ah", "oh", "ah", "ee", "mm"],
        "breathiness": 0.05,
        "jitter": 0.012,
        "shimmer": 0.08,
        "vibrato_rate": 5.0,
        "vibrato_depth": 14,
        "vibrato_delay": 0.25,
        "drift": 6,
        "attack": 0.06,
        "formant_shift": "tenor",
    },
    {
        "name": "bass",
        "type": "voice",
        "octave_range": (2, 3),
        "vowel_sequence": ["oh", "ah", "oh", "oo", "mm"],
        "breathiness": 0.04,
        "jitter": 0.015,
        "shimmer": 0.09,
        "vibrato_rate": 4.5,
        "vibrato_depth": 12,
        "vibrato_delay": 0.3,
        "drift": 7,
        "attack": 0.12,
        "formant_shift": "bass",
    },
]


# =====================================================================
# INSTRUMENT TIMBRES — upgraded with body resonance
# =====================================================================
BODY_RESONANCES = {
    "cello":     [(280, 50), (500, 60), (700, 70), (1200, 100)],
    "clarinet":  [(350, 40), (900, 80), (1500, 100), (2800, 120)],
    "flute":     [(800, 100), (1600, 120), (3200, 140)],
    "oboe":      [(1000, 60), (1800, 80), (2800, 100), (3500, 120)],
    "bell":      [(600, 40), (1200, 60), (2400, 80), (4800, 100)],
    "glass":     [(500, 50), (1500, 80), (3000, 100)],
    "pizz":      [(300, 60), (700, 80), (1400, 100)],
}

# Map instrument timbres to their body resonance profiles
INST_BODY_MAP = {
    "cello_pont": "cello",
    "cello_tasto": "cello",
    "flute_breathy": "flute",
    "clarinet_chalumeau": "clarinet",
    "bell_struck": "bell",
    "glass_harmonica": "glass",
    "pizzicato": "pizz",
    "oboe_pp": "oboe",
}


def apply_body_resonance(signal, resonances, sample_rate, strength=0.3):
    """Apply instrument body resonance via parallel bandpass filters.

    This is what makes a violin sound like a violin vs a speaker cone.
    The body has resonant frequencies that color everything passing through it.
    """
    if not resonances:
        return signal

    body_response = np.zeros_like(signal)

    for f_center, f_bw in resonances:
        if f_center >= sample_rate / 2:
            continue
        w0 = 2 * np.pi * f_center / sample_rate
        Q = f_center / max(f_bw, 1)
        alpha = np.sin(w0) / (2 * Q)

        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]

        body_response += lfilter(b, a, signal)

    # Normalize body response
    peak = np.max(np.abs(body_response))
    if peak > 0:
        body_response /= peak
        body_response *= np.max(np.abs(signal))

    # Mix: mostly original + body coloring
    return signal * (1 - strength) + body_response * strength


def glottal_source(freq_array, n_samples, jitter, rng):
    """Generate glottal pulse train — the voice source signal.

    Models vocal fold vibration using a Rosenberg-type pulse:
      - Open phase: sine-squared rise (~60% of cycle)
      - Closed phase: near-zero (~40% of cycle)

    This produces a naturally rich harmonic spectrum that falls off
    approximately -12dB/octave, like a real voice.

    jitter: cycle-to-cycle frequency perturbation (0-0.03 typical)
    """
    # Add jitter: small random perturbation to instantaneous frequency
    if jitter > 0:
        jitter_noise = 1.0 + jitter * rng.randn(n_samples)
        # Smooth the jitter so it doesn't cause discontinuities
        kernel_size = max(int(SAMPLE_RATE * 0.002), 3)
        kernel = np.ones(kernel_size) / kernel_size
        jitter_noise = np.convolve(jitter_noise, kernel, mode='same')
        freq_jittered = freq_array * jitter_noise
    else:
        freq_jittered = freq_array

    # Accumulate phase from instantaneous frequency
    phase = np.cumsum(2 * np.pi * freq_jittered / SAMPLE_RATE)

    # Normalized position within each cycle (0..1)
    cycle_pos = (phase % (2 * np.pi)) / (2 * np.pi)

    # Rosenberg glottal pulse with smooth transitions
    open_quotient = 0.58  # open phase fraction

    # Main pulse: sine-squared during open phase
    pulse = np.where(
        cycle_pos < open_quotient,
        np.sin(np.pi * cycle_pos / open_quotient) ** 2,
        0.0
    )

    # Smooth the open→closed transition to prevent discontinuities
    # Small raised-cosine crossfade at the boundary
    transition_width = 0.04  # 4% of cycle
    trans_zone = (cycle_pos >= open_quotient - transition_width) & (cycle_pos < open_quotient + transition_width)
    trans_pos = (cycle_pos[trans_zone] - (open_quotient - transition_width)) / (2 * transition_width)
    pulse[trans_zone] *= 0.5 * (1 + np.cos(np.pi * trans_pos))

    # Remove DC offset
    pulse -= np.mean(pulse)

    return pulse


def formant_filter(signal, formant_set, sample_rate):
    """Apply formant resonances to shape the spectral envelope.

    Each formant is a bandpass resonance at a characteristic frequency.
    Together they define the vowel identity.
    """
    output = np.zeros_like(signal)

    for f_freq, f_bw, f_amp in formant_set:
        if f_freq >= sample_rate / 2 - 100:
            continue

        w0 = 2 * np.pi * f_freq / sample_rate
        Q = f_freq / max(f_bw, 1)
        alpha = np.sin(w0) / (2 * Q)

        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]

        output += f_amp * lfilter(b, a, signal)

    return output


def interpolate_formants(vowel_a, vowel_b, t_blend):
    """Linearly interpolate between two vowel formant sets."""
    min_len = min(len(vowel_a), len(vowel_b))
    result = []
    for i in range(min_len):
        f = vowel_a[i][0] * (1 - t_blend) + vowel_b[i][0] * t_blend
        bw = vowel_a[i][1] * (1 - t_blend) + vowel_b[i][1] * t_blend
        amp = vowel_a[i][2] * (1 - t_blend) + vowel_b[i][2] * t_blend
        result.append((f, bw, amp))
    return result


def vocal_tone(t, start, pitch_class, octave, duration, amplitude,
               voice_timbre, rng):
    """Generate a vocal tone with formant synthesis.

    Signal chain:
      1. Glottal pulse source (harmonically rich)
      2. Pitch micro-drift + vibrato (same as instruments)
      3. Formant filtering (vowel shapes, morphing over time)
      4. Breathiness (aspiration noise through formants)
      5. Shimmer (cycle-to-cycle amplitude variation)
      6. Envelope with natural attack/release
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = freq_from_pitch_class(pitch_class, octave)

    if freq < 60 or freq > 1400:  # human vocal range
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)

    # --- PITCH: drift + vibrato ---
    max_drift = voice_timbre["drift"]
    n_drift_pts = max(int(duration * 20), 4)
    drift_walk = np.cumsum(rng.randn(n_drift_pts) * 0.3)
    drift_walk = np.clip(drift_walk, -max_drift, max_drift)
    drift_cents = np.interp(
        np.linspace(0, 1, n_active),
        np.linspace(0, 1, n_drift_pts),
        drift_walk
    )
    freq_base = freq * (2 ** (drift_cents / 1200))

    # Vibrato
    vib_delay = voice_timbre["vibrato_delay"]
    vib_rate = voice_timbre["vibrato_rate"]
    vib_depth = voice_timbre["vibrato_depth"]

    if vib_delay < duration and vib_rate > 0:
        vib_onset = np.clip((t_local - vib_delay) / 0.4, 0, 1)
        rate_wobble = 1.0 + 0.06 * np.sin(2 * np.pi * 0.25 * t_local)
        vibrato = vib_onset * vib_depth * np.sin(
            2 * np.pi * vib_rate * rate_wobble * t_local
        )
        freq_final = freq_base * (2 ** (vibrato / 1200))
    else:
        freq_final = freq_base

    # --- GLOTTAL SOURCE ---
    source = glottal_source(
        freq_final, n_active,
        jitter=voice_timbre["jitter"], rng=rng
    )

    # --- FORMANT FILTERING with vowel morphing ---
    # Uses overlap-add with Hann windowing to eliminate chunk boundary crackles.
    # Each chunk is windowed, formant-filtered at the interpolated vowel position,
    # then summed — Hann 50% overlap gives perfect reconstruction.
    vowel_seq = voice_timbre["vowel_sequence"]
    shift_mult = VOICE_FORMANT_SHIFT[voice_timbre["formant_shift"]]
    n_vowels = len(vowel_seq)

    # Chunk parameters: ~50ms chunks, 50% overlap
    hop_size = max(int(0.05 * SAMPLE_RATE), 256)
    chunk_len = hop_size * 2  # 100% overlap = 2x hop
    window = np.hanning(chunk_len)

    filtered = np.zeros(n_active)
    norm_env = np.zeros(n_active)  # tracks window sum for normalization

    n_chunks = (n_active - chunk_len) // hop_size + 1
    for chunk_idx in range(max(n_chunks, 1)):
        c_start = chunk_idx * hop_size
        c_end = min(c_start + chunk_len, n_active)
        actual_len = c_end - c_start

        if actual_len < 64:
            break

        # Extract and window the source chunk
        chunk_signal = source[c_start:c_end].copy()
        win = window[:actual_len]
        chunk_signal *= win

        # Vowel blend position for this chunk's center
        center = c_start + actual_len // 2
        pos = (center / max(n_active - 1, 1)) * (n_vowels - 1)
        idx_a = min(int(pos), n_vowels - 2)
        idx_b = idx_a + 1
        blend = pos - idx_a

        vowel_a = VOWELS[vowel_seq[idx_a]]
        vowel_b = VOWELS[vowel_seq[idx_b]]
        blended = interpolate_formants(vowel_a, vowel_b, blend)

        # Apply voice-type formant shift
        shifted = [(f * shift_mult, bw, amp) for f, bw, amp in blended]

        chunk_filtered = formant_filter(chunk_signal, shifted, SAMPLE_RATE)

        # Overlap-add
        filtered[c_start:c_end] += chunk_filtered
        norm_env[c_start:c_end] += win

    # Normalize by window sum (prevents amplitude modulation from windowing)
    norm_env = np.maximum(norm_env, 1e-8)
    filtered /= norm_env

    # Normalize filtered signal
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak

    # --- BREATHINESS: aspiration noise through formants ---
    breathiness = voice_timbre["breathiness"]
    if breathiness > 0:
        breath_noise = rng.randn(n_active)
        # Shape breath noise through the current vowel formants
        mid_vowel = VOWELS[vowel_seq[len(vowel_seq) // 2]]
        mid_shifted = [(f * shift_mult, bw * 1.5, amp) for f, bw, amp in mid_vowel]
        breath_filtered = formant_filter(breath_noise, mid_shifted, SAMPLE_RATE)
        b_peak = np.max(np.abs(breath_filtered))
        if b_peak > 0:
            breath_filtered /= b_peak

        # Breath envelope: more at start and end, less in middle
        breath_env = 0.4 + 0.6 * (1.0 - np.sin(t_norm * np.pi) * 0.6)
        filtered += breathiness * breath_filtered * breath_env

    # --- SHIMMER: amplitude perturbation ---
    shimmer = voice_timbre["shimmer"]
    if shimmer > 0:
        # Slow amplitude variation (~matching vibrato rate)
        shimmer_mod = 1.0 + shimmer * rng.randn(n_active)
        kernel_size = max(int(SAMPLE_RATE * 0.005), 3)
        kernel = np.ones(kernel_size) / kernel_size
        shimmer_mod = np.convolve(shimmer_mod, kernel, mode='same')
        filtered *= np.clip(shimmer_mod, 0.7, 1.3)

    # --- ENVELOPE ---
    attack_time = voice_timbre["attack"]
    attack_samples = max(int(attack_time * SAMPLE_RATE), 1)

    envelope = np.ones(n_active)

    # Vocal attack: gradual onset (not percussive)
    if attack_samples < n_active:
        att = np.linspace(0, 1, min(attack_samples, n_active))
        att = att ** 1.5  # slightly curved — not linear
        envelope[:len(att)] = att

    # Sustain with gentle taper
    envelope *= np.maximum(0, 1.0 - t_norm * 0.15)
    envelope *= np.exp(-0.4 * t_norm)

    # Release: last 8% of note fades smoothly
    release_start = int(n_active * 0.92)
    if release_start < n_active:
        release_len = n_active - release_start
        envelope[release_start:] *= np.linspace(1, 0, release_len) ** 1.5

    # Anti-click
    fadeout = min(int(0.02 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        envelope[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = filtered * envelope * amplitude
    return voice


def instrument_tone_with_body(t, start, pitch_class, octave, duration,
                               amplitude, timbre, rng):
    """Instrument tone with body resonance coloring.

    Uses the original synthesis from webern_pointillism but adds
    resonant body filtering for more realistic timbre.
    """
    from webern_pointillism import pointillist_tone

    raw = pointillist_tone(t, start, pitch_class, octave, duration,
                           amplitude, timbre, rng)

    # Find the body resonance for this instrument
    timbre_name = timbre["name"]
    body_key = INST_BODY_MAP.get(timbre_name)

    if body_key and body_key in BODY_RESONANCES:
        resonances = BODY_RESONANCES[body_key]
        # Extract just the active region for efficiency
        mask = (t >= start) & (t < start + duration + 0.5)
        if np.any(mask):
            region = raw[mask].copy()
            region = apply_body_resonance(region, resonances, SAMPLE_RATE, strength=0.25)
            raw[mask] = region

    return raw


# =====================================================================
# BERG ROW — same triadic row
# =====================================================================
BERG_ROW = np.array([7, 10, 2, 6, 9, 0, 4, 8, 11, 1, 3, 5])

def row_transpose(row, interval):
    return (row + interval) % 12

def row_inversion(row):
    return (2 * row[0] - row) % 12

def row_retrograde(row):
    return row[::-1]


def make_vocal_line(rng, row_form, start_time, duration, voice_timbre,
                    amp_base, amp_peak):
    """Generate a vocal melodic line — Berg's lyrical singing voice."""
    events = []
    cursor = start_time
    oct_low, oct_high = voice_timbre["octave_range"]

    pitches = list(row_form)
    while cursor < start_time + duration:
        for pc in pitches:
            if cursor >= start_time + duration:
                break

            # Vocal tones: moderate to long, always legato
            tone_dur = rng.uniform(2.5, 6.0)

            # Stay in vocal range with occasional range extension
            octave = rng.choice([oct_low, oct_low, oct_high, oct_low])

            # Expressionist dynamic arc
            phrase_pos = (cursor - start_time) / max(duration, 1)
            arc = np.sin(phrase_pos * np.pi)
            amp = amp_base + (amp_peak - amp_base) * arc
            amp *= rng.uniform(0.9, 1.1)

            events.append({
                "time": cursor,
                "pc": int(pc),
                "octave": octave,
                "duration": tone_dur,
                "amplitude": min(amp, 0.30),
                "voice_timbre": voice_timbre,
                "type": "voice"
            })

            # Legato overlap
            advance = tone_dur * rng.uniform(0.5, 0.75)
            cursor += advance

    return events


def make_inst_line(rng, row_form, start_time, duration, octave_center,
                   amp_base, amp_peak, timbre_pool):
    """Generate an instrumental line with body resonance."""
    events = []
    cursor = start_time

    pitches = list(row_form)
    while cursor < start_time + duration:
        for pc in pitches:
            if cursor >= start_time + duration:
                break

            tone_dur = rng.uniform(2.0, 5.0)
            octave = octave_center + rng.choice([-1, 0, 0, 0, 1])
            octave = max(2, min(6, octave))

            phrase_pos = (cursor - start_time) / max(duration, 1)
            arc = np.sin(phrase_pos * np.pi)
            amp = amp_base + (amp_peak - amp_base) * arc
            amp *= rng.uniform(0.85, 1.15)

            events.append({
                "time": cursor,
                "pc": int(pc),
                "octave": octave,
                "duration": tone_dur,
                "amplitude": min(amp, 0.30),
                "timbre": rng.choice(timbre_pool),
                "type": "instrument"
            })

            advance = tone_dur * rng.uniform(0.45, 0.8)
            cursor += advance

    return events


def generate_berg_vocal(seed=None):
    """Generate a Berg choral-orchestral piece."""

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

    soprano = VOICE_TIMBRES[0]
    alto = VOICE_TIMBRES[1]
    tenor = VOICE_TIMBRES[2]
    bass = VOICE_TIMBRES[3]

    strings = [INST_TIMBRES[0], INST_TIMBRES[1]]
    winds = [INST_TIMBRES[2], INST_TIMBRES[3], INST_TIMBRES[7]]
    colors = [INST_TIMBRES[4], INST_TIMBRES[5]]

    all_events = []

    # ================================================================
    # SECTION 1: Instrumental intro + bass voice emerges (0-25s)
    # ================================================================
    print("  Section 1: Strings intro, bass voice enters (0-25s)")

    all_events.extend(make_inst_line(
        rng, row_P, 0.5, 24, octave_center=3,
        amp_base=0.05, amp_peak=0.12, timbre_pool=strings
    ))

    # Bass enters at 8s — emerging from the strings
    all_events.extend(make_vocal_line(
        rng, row_P, 8.0, 16, bass,
        amp_base=0.06, amp_peak=0.16
    ))

    # ================================================================
    # SECTION 2: Voices accumulate — tenor, then alto (20-50s)
    # ================================================================
    print("  Section 2: Voices accumulate — tenor, alto join (20-50s)")

    # Tenor enters with inverted row
    all_events.extend(make_vocal_line(
        rng, row_I, 20.0, 28, tenor,
        amp_base=0.07, amp_peak=0.18
    ))

    # Alto enters at 28s with retrograde
    all_events.extend(make_vocal_line(
        rng, row_R, 28.0, 20, alto,
        amp_base=0.06, amp_peak=0.16
    ))

    # Winds weave underneath
    all_events.extend(make_inst_line(
        rng, row_transpose(row_P, 5), 22.0, 25, octave_center=4,
        amp_base=0.04, amp_peak=0.10, timbre_pool=winds
    ))

    # ================================================================
    # SECTION 3: Full choir + soprano — climax (45-75s)
    # ================================================================
    print("  Section 3: Full choir + soprano — climax (45-75s)")

    # Soprano enters — the dramatic peak
    all_events.extend(make_vocal_line(
        rng, row_P, 45.0, 28, soprano,
        amp_base=0.08, amp_peak=0.25
    ))

    # All other voices continue
    all_events.extend(make_vocal_line(
        rng, row_transpose(row_I, 3), 48.0, 22, alto,
        amp_base=0.07, amp_peak=0.18
    ))
    all_events.extend(make_vocal_line(
        rng, row_transpose(row_P, 7), 46.0, 24, tenor,
        amp_base=0.07, amp_peak=0.20
    ))
    all_events.extend(make_vocal_line(
        rng, row_R, 50.0, 18, bass,
        amp_base=0.06, amp_peak=0.16
    ))

    # Full orchestra underneath
    all_events.extend(make_inst_line(
        rng, row_P, 48.0, 20, octave_center=3,
        amp_base=0.05, amp_peak=0.14, timbre_pool=strings
    ))
    all_events.extend(make_inst_line(
        rng, row_I, 50.0, 18, octave_center=5,
        amp_base=0.03, amp_peak=0.08, timbre_pool=colors
    ))

    # ================================================================
    # SECTION 4: Tonal window — G minor, voices in unison/octaves (72-85s)
    # ================================================================
    print("  Section 4: Tonal window — Gm chorale (72-85s)")

    # G minor triad sustained by all four voices
    chorale_pcs = [7, 10, 2, 7]  # G, Bb, D, G
    chorale_voices = [bass, tenor, alto, soprano]
    chorale_octs = [2, 3, 4, 5]

    for pc, v, o in zip(chorale_pcs, chorale_voices, chorale_octs):
        all_events.append({
            "time": 72.0 + rng.uniform(0, 0.5),
            "pc": pc,
            "octave": o,
            "duration": rng.uniform(7.0, 10.0),
            "amplitude": 0.14,
            "voice_timbre": v,
            "type": "voice"
        })

    # Add the 7th and passing tones — Berg wouldn't let it stay pure
    for _ in range(3):
        all_events.append({
            "time": rng.uniform(74, 80),
            "pc": rng.choice([5, 1, 6]),  # F, C#, F# — chromatic color
            "octave": rng.choice([4, 5]),
            "duration": rng.uniform(3.0, 5.0),
            "amplitude": rng.uniform(0.04, 0.08),
            "voice_timbre": rng.choice([soprano, alto]),
            "type": "voice"
        })

    # Sustained strings underneath chorale
    for pc in [7, 2]:
        all_events.append({
            "time": 72.5,
            "pc": pc,
            "octave": 3,
            "duration": 10.0,
            "amplitude": 0.08,
            "timbre": INST_TIMBRES[1],  # cello tasto
            "type": "instrument"
        })

    # ================================================================
    # SECTION 5: Dissolution (83-98s)
    # ================================================================
    print("  Section 5: Dissolution — voices fade (83-98s)")

    # Solo soprano, very quiet, fragments of the row
    all_events.extend(make_vocal_line(
        rng, row_R, 83.0, 12, soprano,
        amp_base=0.03, amp_peak=0.08
    ))

    # Bass hum underneath
    all_events.append({
        "time": 85.0,
        "pc": 7,  # G — tonal anchor to the end
        "octave": 2,
        "duration": 10.0,
        "amplitude": 0.05,
        "voice_timbre": bass,
        "type": "voice"
    })

    # Glass/bell color tones evaporating
    for i in range(4):
        all_events.append({
            "time": 86.0 + i * 2.5,
            "pc": int(row_P[i]),
            "octave": rng.choice([5, 6]),
            "duration": rng.uniform(2.0, 4.0),
            "amplitude": rng.uniform(0.02, 0.04),
            "timbre": rng.choice(colors),
            "type": "instrument"
        })

    # ================================================================
    # RENDER
    # ================================================================
    n_voices = sum(1 for e in all_events if e["type"] == "voice")
    n_inst = sum(1 for e in all_events if e["type"] == "instrument")
    print(f"\n  Rendering {len(all_events)} events ({n_voices} vocal, {n_inst} instrumental)...")

    for event in all_events:
        if event["type"] == "voice":
            audio += vocal_tone(
                t, event["time"], event["pc"], event["octave"],
                event["duration"], event["amplitude"],
                event["voice_timbre"], np_rng
            )
        else:
            audio += instrument_tone_with_body(
                t, event["time"], event["pc"], event["octave"],
                event["duration"], event["amplitude"],
                event["timbre"], np_rng
            )

    # Reverb — concert hall wet
    audio = simple_reverb(audio, decay=0.5, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 2.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 3.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.08)

    audio = np.tanh(audio * 1.08) / 1.08

    return audio, all_events


def main():
    parser = argparse.ArgumentParser(description="Berg vocal synthesis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="berg_vocal_01.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Generating Berg choral-orchestral piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Voices: soprano, alto, tenor, bass\n")

    audio, events = generate_berg_vocal(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")
    print(f"  Sections: Intro → Voices accumulate → Climax → Chorale → Dissolution")


if __name__ == "__main__":
    main()
