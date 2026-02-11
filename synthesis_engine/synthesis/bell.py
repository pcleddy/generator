"""
bell.py — Bell synthesis: proper inharmonic bell partials.

Extracted from bells_bergman.py. Also handles wood percussion (same structure).
Accepts profile as either a string (key into BELL_PROFILES) or a dict.
"""

import numpy as np
from scipy.signal import lfilter

from ..config import SAMPLE_RATE
from ..profiles.bells import BELL_PROFILES


def bell_strike(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a single bell strike with proper bell acoustics.

    Bell partials are NOT harmonic — they follow specific ratios
    determined by the bell's geometry and material.

    Args:
        profile: string key into BELL_PROFILES, or a dict with the same structure.
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


def wood_strike(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a wooden bar being struck (xylophone, marimba).

    Similar to bell_strike but warmer attack, shorter sustain,
    resonator body hum underneath.
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)

    if isinstance(profile, str):
        from ..profiles.wood import WOOD_PROFILES
        profile = WOOD_PROFILES[profile]

    if freq < 20 or freq > 8000:
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

        detune = 1.0 + rng.uniform(-0.001, 0.001)
        partial_freq *= detune
        phase = 2 * np.pi * partial_freq * t_local + rng.uniform(0, 2 * np.pi)

        effective_decay = decay_rate / profile["ring_time_mult"]
        envelope = np.exp(-effective_decay * t_local)

        signal += partial_amp * np.sin(phase) * envelope

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # Wood strike — warm bandpass
    strike_noise = rng.randn(n_active)
    center = profile["strike_brightness"]
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = 1.5  # wider Q — softer attack
    alpha = np.sin(w0) / (2 * Q)
    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    strike_noise = lfilter(b, a, strike_noise)

    strike_len = min(int(0.012 * SAMPLE_RATE), n_active)
    strike_env = np.zeros(n_active)
    if strike_len > 0:
        strike_env[:strike_len] = np.exp(-np.linspace(0, 10, strike_len))

    strike = strike_noise * strike_env
    s_peak = np.max(np.abs(strike))
    if s_peak > 0:
        strike /= s_peak

    combined = signal * 0.88 + strike * profile["strike_amount"]

    # Resonator body hum
    res_phase = 2 * np.pi * freq * t_local + rng.uniform(0, 2 * np.pi)
    res_signal = 0.08 * np.sin(res_phase) * np.exp(-1.5 * t_local)
    combined += res_signal

    overall_env = np.ones(n_active)
    attack_len = max(int(0.003 * SAMPLE_RATE), 1)
    if attack_len < n_active:
        overall_env[:attack_len] = np.linspace(0, 1, attack_len)
    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        overall_env[-fadeout:] *= np.linspace(1, 0, fadeout)

    voice[mask] = combined * overall_env * amplitude
    return voice
