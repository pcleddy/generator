"""
one_more_thing.py — Grand finale

Ambient bed with Paul's cloned voice emerging from the texture.
Voice synthesis tuned to Paul's measured formant profile from
his iPad voice memo recording.

Measured characteristics:
  F0:  92 Hz (bass-baritone speaking range)
  F1: 673 Hz  (BW: 155 Hz)
  F2: 996 Hz  (BW:  39 Hz) — remarkably tight
  F3: 3049 Hz (BW:  91 Hz)
  F4: 4780 Hz (BW: 257 Hz)

FIX: Spectral tilt reduced and glottal pulse sharpened so upper
harmonics survive to feed F2/F3/F4 formant resonators.

Structure:
  0-20s:   Ambient drift, sub-bass, stillness
  18-45s:  Paul's voice emerges — low hums, then vowel shapes
  40-70s:  Voice and instruments converse — call and response
  65-85s:  Full texture: voice + ambient + prepared piano punctuation
  80-100s: Voice alone, fading into silence

Usage:
    python one_more_thing.py [--seed N] [--output filename.wav]
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
from berg_vocal import (
    VOWELS, VOICE_FORMANT_SHIFT, vocal_tone, interpolate_formants,
    VOICE_TIMBRES, apply_body_resonance, BODY_RESONANCES,
    instrument_tone_with_body
)
from cage_ambient import prepared_piano_strike, tone_cluster

DURATION = 105


# =====================================================================
# PAUL'S VOICE — cloned from spectral analysis
# =====================================================================

# Custom formant data measured from Paul's recording
# Note: using measured bandwidths, not generic ones
PAUL_FORMANTS = {
    "paul_ah": [
        (673,  155, 1.0),     # F1 — measured
        (996,   50, 0.70),    # F2 — measured (tightened slightly from 39)
        (3049,  91, 0.20),    # F3 — measured
        (4780, 257, 0.06),    # F4 — measured
    ],
    "paul_oh": [
        (500,  120, 1.0),     # F1 — shifted down from ah
        (850,   60, 0.60),    # F2
        (2900, 100, 0.15),    # F3
        (4600, 250, 0.05),    # F4
    ],
    "paul_mm": [
        (280,   80, 1.0),     # F1 — nasal
        (900,   70, 0.20),    # F2 — reduced (closed mouth)
        (2800, 110, 0.06),    # F3
        (4500, 260, 0.02),    # F4
    ],
    "paul_eh": [
        (580,  130, 1.0),     # F1
        (1050,  55, 0.65),    # F2
        (3100, 100, 0.18),    # F3
        (4800, 250, 0.05),    # F4
    ],
}

# Add Paul's vowels to the VOWELS dict for the formant engine
VOWELS.update(PAUL_FORMANTS)

PAUL_VOICE = {
    "name": "paul_clone",
    "type": "voice",
    "octave_range": (2, 3),       # bass-baritone
    "vowel_sequence": ["paul_mm", "paul_ah", "paul_eh", "paul_oh", "paul_mm"],
    "breathiness": 0.04,           # moderate
    "jitter": 0.012,               # natural male voice jitter
    "shimmer": 0.08,
    "vibrato_rate": 4.8,           # relaxed male vibrato
    "vibrato_depth": 10,           # moderate
    "vibrato_delay": 0.4,          # slow onset
    "drift": 6,                    # natural pitch wander
    "attack": 0.10,                # gentle onset
    "formant_shift": "paul",       # custom shift factor
}

# Paul's voice uses no formant shift — formants ARE his measured values
VOICE_FORMANT_SHIFT["paul"] = 1.0


# =====================================================================
# FIXED GLOTTAL SOURCE — reduced spectral tilt
# =====================================================================

def glottal_source_bright(freq_array, n_samples, jitter, rng):
    """Glottal pulse with REDUCED spectral tilt.

    The original had tilt_coeff=0.96 which killed upper harmonics.
    Spectral comparison showed our synthesis dying above 1kHz while
    Paul's real voice has strong harmonics to 3.5kHz+.

    Fix: reduce tilt to 0.7 (much less rolloff) and use a sharper
    open quotient (0.45 instead of 0.58) for richer harmonics.
    """
    # Jitter
    if jitter > 0:
        jitter_noise = 1.0 + jitter * rng.randn(n_samples)
        kernel_size = max(int(SAMPLE_RATE * 0.002), 3)
        kernel = np.ones(kernel_size) / kernel_size
        jitter_noise = np.convolve(jitter_noise, kernel, mode='same')
        freq_jittered = freq_array * jitter_noise
    else:
        freq_jittered = freq_array

    phase = np.cumsum(2 * np.pi * freq_jittered / SAMPLE_RATE)
    cycle_pos = (phase % (2 * np.pi)) / (2 * np.pi)

    # SHARPER open quotient = richer harmonics
    open_quotient = 0.45  # was 0.58 — shorter open phase = brighter

    pulse = np.where(
        cycle_pos < open_quotient,
        np.sin(np.pi * cycle_pos / open_quotient) ** 2,
        0.0
    )

    # Smooth transition
    tw = 0.04
    trans_zone = (cycle_pos >= open_quotient - tw) & (cycle_pos < open_quotient + tw)
    trans_pos = (cycle_pos[trans_zone] - (open_quotient - tw)) / (2 * tw)
    pulse[trans_zone] *= 0.5 * (1 + np.cos(np.pi * trans_pos))

    pulse -= np.mean(pulse)

    # REDUCED spectral tilt — let upper harmonics live
    tilt_coeff = 0.7  # was 0.96
    pulse = lfilter([1.0], [1.0, -tilt_coeff], pulse)
    pulse -= np.mean(pulse)

    return pulse


def paul_vocal_tone(t, start, pitch_class, octave, duration, amplitude,
                    voice_timbre, rng):
    """Vocal tone using Paul's formant profile and fixed glottal source.

    Same signal chain as vocal_tone but with:
    - Brighter glottal source (reduced tilt, sharper OQ)
    - Paul's measured formant data
    """
    from berg_vocal import (
        formant_filter, freq_from_pitch_class,
    )

    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = freq_from_pitch_class(pitch_class, octave)

    if freq < 50 or freq > 500:  # Paul's speaking range
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)

    # Pitch drift
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

    # Irregular vibrato
    vib_delay = voice_timbre["vibrato_delay"]
    vib_rate = voice_timbre["vibrato_rate"]
    vib_depth = voice_timbre["vibrato_depth"]

    if vib_delay < duration and vib_rate > 0:
        vib_onset = np.clip((t_local - vib_delay) / 0.5, 0, 1) ** 1.5
        n_rate_pts = max(int(duration * 8), 4)
        rate_noise = np.cumsum(rng.randn(n_rate_pts) * 0.15)
        rate_noise = np.clip(rate_noise, -0.4, 0.4)
        rate_mod = np.interp(np.linspace(0, 1, n_active),
                              np.linspace(0, 1, n_rate_pts), rate_noise)
        inst_rate = vib_rate * (1.0 + rate_mod * 0.15)

        n_depth_pts = max(int(duration * 6), 4)
        depth_noise = np.cumsum(rng.randn(n_depth_pts) * 0.2)
        depth_noise = np.clip(depth_noise, -0.6, 0.6)
        depth_mod = np.interp(np.linspace(0, 1, n_active),
                               np.linspace(0, 1, n_depth_pts), depth_noise)
        inst_depth = vib_depth * (1.0 + depth_mod * 0.25)

        vib_phase = np.cumsum(2 * np.pi * inst_rate / SAMPLE_RATE)
        vibrato = vib_onset * inst_depth * np.sin(vib_phase)
        freq_final = freq_base * (2 ** (vibrato / 1200))
    else:
        freq_final = freq_base

    # BRIGHT glottal source
    source = glottal_source_bright(freq_final, n_active,
                                    jitter=voice_timbre["jitter"], rng=rng)

    # Formant filtering with overlap-add
    vowel_seq = voice_timbre["vowel_sequence"]
    shift_mult = VOICE_FORMANT_SHIFT.get(voice_timbre.get("formant_shift", "alto"), 1.0)
    n_vowels = len(vowel_seq)

    hop_size = max(int(0.05 * SAMPLE_RATE), 256)
    chunk_len = hop_size * 2
    window = np.hanning(chunk_len)

    filtered = np.zeros(n_active)
    norm_env = np.zeros(n_active)

    n_chunks = (n_active - chunk_len) // hop_size + 1
    for chunk_idx in range(max(n_chunks, 1)):
        c_start = chunk_idx * hop_size
        c_end = min(c_start + chunk_len, n_active)
        actual_len = c_end - c_start
        if actual_len < 64:
            break

        chunk_signal = source[c_start:c_end].copy()
        win = window[:actual_len]
        chunk_signal *= win

        center = c_start + actual_len // 2
        pos = (center / max(n_active - 1, 1)) * (n_vowels - 1)
        idx_a = min(int(pos), n_vowels - 2)
        idx_b = idx_a + 1
        blend = pos - idx_a

        vowel_a = VOWELS[vowel_seq[idx_a]]
        vowel_b = VOWELS[vowel_seq[idx_b]]
        blended = interpolate_formants(vowel_a, vowel_b, blend)
        shifted = [(f * shift_mult, bw, amp) for f, bw, amp in blended]

        chunk_filtered = formant_filter(chunk_signal, shifted, SAMPLE_RATE)
        filtered[c_start:c_end] += chunk_filtered
        norm_env[c_start:c_end] += win

    norm_env = np.maximum(norm_env, 1e-8)
    filtered /= norm_env

    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak

    # Light breathiness
    breathiness = voice_timbre["breathiness"]
    if breathiness > 0:
        breath = rng.randn(n_active)
        mid_vowel = VOWELS[vowel_seq[len(vowel_seq) // 2]]
        mid_shifted = [(f * shift_mult, bw * 2.0, amp * 0.5) for f, bw, amp in mid_vowel]
        breath_filtered = formant_filter(breath, mid_shifted, SAMPLE_RATE)
        bp = np.max(np.abs(breath_filtered))
        if bp > 0:
            breath_filtered /= bp
        breath_env = np.exp(-8.0 * t_norm) + 0.1 * np.exp(-3.0 * (1 - t_norm) ** 2)
        filtered += breathiness * 0.4 * breath_filtered * np.clip(breath_env, 0, 1)

    # Shimmer
    shimmer = voice_timbre["shimmer"]
    if shimmer > 0:
        shimmer_mod = 1.0 + shimmer * rng.randn(n_active)
        ks = max(int(SAMPLE_RATE * 0.005), 3)
        kernel = np.ones(ks) / ks
        shimmer_mod = np.convolve(shimmer_mod, kernel, mode='same')
        filtered *= np.clip(shimmer_mod, 0.7, 1.3)

    # Envelope
    attack_time = voice_timbre["attack"]
    attack_samples = max(int(attack_time * SAMPLE_RATE), 1)
    envelope = np.ones(n_active)

    if attack_samples < n_active:
        att = np.linspace(0, 1, min(attack_samples, n_active)) ** 1.5
        envelope[:len(att)] = att

    envelope *= np.maximum(0, 1.0 - t_norm * 0.15)
    envelope *= np.exp(-0.4 * t_norm)

    release_start = int(n_active * 0.90)
    if release_start < n_active:
        envelope[release_start:] *= np.linspace(1, 0, n_active - release_start) ** 1.5

    fadeout = min(int(0.02 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        envelope[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = filtered * envelope * amplitude
    return voice


# =====================================================================
# COMPOSITION
# =====================================================================

BERG_ROW = np.array([7, 10, 2, 6, 9, 0, 4, 8, 11, 1, 3, 5])


def generate_piece(seed=None):
    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)

    # ================================================================
    # SECTION 1: Ambient drift (0-20s)
    # ================================================================
    print("  Section 1: Ambient drift (0-20s)")

    twelve_tone = BERG_ROW
    for i, pitch in enumerate(twelve_tone):
        start = i * 3.5
        if start + 8 > 25:
            break
        timbre = rng.choice([INST_TIMBRES[1], INST_TIMBRES[5]])  # tasto, glass
        audio += instrument_tone_with_body(
            t, start, int(pitch), 3,
            rng.uniform(6, 12), rng.uniform(0.04, 0.06),
            timbre, np_rng
        )

    # Sub-bass anchor
    audio += instrument_tone_with_body(
        t, 0, 7, 2, 20, 0.03, INST_TIMBRES[1], np_rng  # G2 drone
    )

    # ================================================================
    # SECTION 2: Paul's voice emerges (18-45s)
    # ================================================================
    print("  Section 2: Paul's voice emerges (18-45s)")

    # Start with a low hum (paul_mm)
    hum_timbre = PAUL_VOICE.copy()
    hum_timbre["vowel_sequence"] = ["paul_mm", "paul_mm"]
    audio += paul_vocal_tone(
        t, 18.0, 7, 2,  # G2 — close to Paul's F0 of 92 Hz
        duration=8.0, amplitude=0.12,
        voice_timbre=hum_timbre, rng=np_rng
    )

    # Gradually open up: mm → ah
    opening_timbre = PAUL_VOICE.copy()
    opening_timbre["vowel_sequence"] = ["paul_mm", "paul_ah", "paul_ah"]
    audio += paul_vocal_tone(
        t, 27.0, 9, 2,  # A2 — 110 Hz, near Paul's range
        duration=7.0, amplitude=0.18,
        voice_timbre=opening_timbre, rng=np_rng
    )

    # Full voice: ah → eh → oh
    full_timbre = PAUL_VOICE.copy()
    full_timbre["vowel_sequence"] = ["paul_ah", "paul_eh", "paul_oh", "paul_mm"]
    audio += paul_vocal_tone(
        t, 35.0, 7, 2,  # G2
        duration=9.0, amplitude=0.22,
        voice_timbre=full_timbre, rng=np_rng
    )

    # ================================================================
    # SECTION 3: Voice and instruments converse (40-70s)
    # ================================================================
    print("  Section 3: Voice and instruments converse (40-70s)")

    # Instrument responses to voice
    responses = [
        (42.0, 4, 5, INST_TIMBRES[2]),   # flute, high E5
        (47.0, 11, 4, INST_TIMBRES[7]),   # oboe, B4
        (52.0, 2, 5, INST_TIMBRES[5]),    # glass, D5
        (57.0, 8, 4, INST_TIMBRES[3]),    # clarinet, G#4
    ]

    for start, pc, oct, timbre in responses:
        audio += instrument_tone_with_body(
            t, start, pc, oct,
            rng.uniform(4, 7), rng.uniform(0.06, 0.10),
            timbre, np_rng
        )

    # Paul responds back
    for start, pc, vowels in [
        (45.0, 9, ["paul_oh", "paul_ah", "paul_oh"]),
        (50.0, 5, ["paul_ah", "paul_eh", "paul_ah"]),
        (55.0, 7, ["paul_eh", "paul_oh", "paul_mm"]),
        (60.0, 2, ["paul_mm", "paul_ah", "paul_eh", "paul_oh"]),
    ]:
        call_timbre = PAUL_VOICE.copy()
        call_timbre["vowel_sequence"] = vowels
        audio += paul_vocal_tone(
            t, start, pc, 2,
            duration=rng.uniform(5, 8), amplitude=rng.uniform(0.15, 0.22),
            voice_timbre=call_timbre, rng=np_rng
        )

    # ================================================================
    # SECTION 4: Full texture + prepared piano punctuation (65-85s)
    # ================================================================
    print("  Section 4: Full texture with piano interruptions (65-85s)")

    # Paul's voice, sustained
    climax_timbre = PAUL_VOICE.copy()
    climax_timbre["vowel_sequence"] = ["paul_ah", "paul_eh", "paul_ah", "paul_oh", "paul_ah"]
    climax_timbre["vibrato_depth"] = 14  # more intense
    audio += paul_vocal_tone(
        t, 65.0, 7, 2,
        duration=12.0, amplitude=0.25,
        voice_timbre=climax_timbre, rng=np_rng
    )

    # Ambient underneath
    for i, pitch in enumerate(twelve_tone[:6]):
        start = 66.0 + i * 2.5
        timbre = rng.choice([INST_TIMBRES[0], INST_TIMBRES[1]])
        audio += instrument_tone_with_body(
            t, start, int(pitch), 3,
            rng.uniform(5, 8), rng.uniform(0.04, 0.07),
            timbre, np_rng
        )

    # Two prepared piano interruptions
    print("    Piano strike at 70.0s")
    audio += tone_cluster(
        t, 70.0, 7, 3, 10, 12, 2.5, 0.20, "bolt", np_rng
    )

    print("    Piano strike at 78.0s")
    audio += prepared_piano_strike(
        t, 78.0, 2, 3, 3.0, 0.22, "screw", np_rng
    )

    # ================================================================
    # SECTION 5: Voice alone, fading (80-100s)
    # ================================================================
    print("  Section 5: Voice alone, dissolving (80-100s)")

    # Paul humming, fading away
    fade_timbre = PAUL_VOICE.copy()
    fade_timbre["vowel_sequence"] = ["paul_oh", "paul_mm", "paul_mm"]
    fade_timbre["breathiness"] = 0.06  # more breath as voice tires

    audio += paul_vocal_tone(
        t, 82.0, 7, 2,  # back to G2
        duration=10.0, amplitude=0.14,
        voice_timbre=fade_timbre, rng=np_rng
    )

    # One last breath
    final_timbre = PAUL_VOICE.copy()
    final_timbre["vowel_sequence"] = ["paul_mm", "paul_mm"]
    final_timbre["breathiness"] = 0.10
    audio += paul_vocal_tone(
        t, 93.0, 7, 2,
        duration=6.0, amplitude=0.06,
        voice_timbre=final_timbre, rng=np_rng
    )

    # Single glass tone — last sound
    audio += instrument_tone_with_body(
        t, 96.0, 7, 5, 5.0, 0.03, INST_TIMBRES[5], np_rng
    )

    # ================================================================
    # MIX
    # ================================================================
    # Wetter reverb — this is intimate
    audio = simple_reverb(audio, decay=0.5, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 3.0, 1.0)
    fade_out = np.minimum((DURATION - t) / 4.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.08)

    audio = np.tanh(audio * 1.05) / 1.05

    return audio


def main():
    parser = argparse.ArgumentParser(description="One More Thing — Paul's voice clone")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="one_more_thing.wav",
                        help="Output filename")
    args = parser.parse_args()

    print(f'Generating "One More Thing"...')
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Voice: Paul clone (F0≈92Hz, F1=673, F2=996, F3=3049)\n")

    audio = generate_piece(seed=args.seed)

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"\nGenerated: {args.output}")


if __name__ == "__main__":
    main()
