"""
tone.py — Core pitched tone synthesis (strings, winds, etc.)

The pointillist_tone() function from webern_pointillism.py.
"""

import numpy as np
from ..config import SAMPLE_RATE, freq_from_pitch_class
from .envelopes import generate_noise


def pointillist_tone(t, start, pitch_class, octave, duration, amplitude,
                     timbre, rng):
    """Generate a single tone with organic synthesis modeling.

    Key features:
      1. Per-partial envelopes: upper harmonics decay faster
      2. Attack transient: filtered noise burst at onset
      3. Pitch micro-drift: slow random walk
      4. Delayed vibrato: ramps in like a real player
      5. Slight inharmonicity: partials aren't perfect ratios
      6. Optional undertone (subharmonic)
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = freq_from_pitch_class(pitch_class, octave)

    if freq < 30 or freq > 9000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = int(np.sum(mask))
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)

    # --- 1. PITCH MICRO-DRIFT ---
    max_drift_cents = timbre.get("drift", 5)
    n_drift_points = max(int(duration * 20), 4)
    drift_walk = np.cumsum(rng.randn(n_drift_points) * 0.3)
    drift_walk = np.clip(drift_walk, -max_drift_cents, max_drift_cents)
    drift_cents = np.interp(
        np.linspace(0, 1, n_active),
        np.linspace(0, 1, n_drift_points),
        drift_walk
    )
    freq_drift = freq * (2 ** (drift_cents / 1200))

    # --- 2. DELAYED VIBRATO ---
    vib_delay = timbre.get("vibrato_delay", 99)
    vib_rate = timbre.get("vibrato_rate", 0)
    vib_depth_cents = timbre.get("vibrato_depth", 0)

    if vib_delay < duration and vib_rate > 0:
        vib_onset = np.clip((t_local - vib_delay) / 0.3, 0, 1)
        rate_wobble = 1.0 + 0.05 * np.sin(2 * np.pi * 0.3 * t_local)
        vibrato = vib_onset * vib_depth_cents * np.sin(
            2 * np.pi * vib_rate * rate_wobble * t_local
        )
        freq_with_vib = freq_drift * (2 ** (vibrato / 1200))
    else:
        freq_with_vib = freq_drift

    # --- 3. PER-PARTIAL SYNTHESIS ---
    signal = np.zeros(n_active)
    base_harmonics = timbre.get("harmonics", [1.0])
    partial_decay_rate = timbre.get("partial_decay", 1.0)

    for h_num, h_amp in enumerate(base_harmonics, 1):
        if h_amp < 0.005:
            continue

        stiffness = rng.uniform(0.0001, 0.0004)
        inharmonic_ratio = h_num * np.sqrt(1 + stiffness * h_num * h_num)

        inst_freq = freq_with_vib * inharmonic_ratio
        phase = np.cumsum(2 * np.pi * inst_freq / SAMPLE_RATE)
        phase += rng.uniform(0, 2 * np.pi)

        partial_signal = h_amp * np.sin(phase)

        if timbre.get("decay_shape") == "exp":
            decay_rate = 3.0 + (h_num - 1) * partial_decay_rate
            partial_env = np.exp(-decay_rate * t_norm)
        else:
            base_sustain = np.maximum(0, 1.0 - t_norm * 0.3)
            upper_fade = np.exp(-(h_num - 1) * partial_decay_rate * 0.5 * t_norm)
            partial_env = base_sustain * upper_fade

        signal += partial_signal * partial_env

    total_weight = sum(h for h in base_harmonics if h >= 0.005)
    if total_weight > 0:
        signal /= total_weight

    # --- 4. UNDERTONE ---
    undertone_amp = timbre.get("undertone", 0)
    if undertone_amp > 0:
        sub_phase = np.cumsum(2 * np.pi * (freq_with_vib * 0.5) / SAMPLE_RATE)
        sub_phase += rng.uniform(0, 2 * np.pi)
        sub_env = np.minimum(t_norm * 3, 1.0) * np.exp(-1.5 * t_norm)
        signal += undertone_amp * np.sin(sub_phase) * sub_env

    # --- 5. ATTACK ENVELOPE ---
    attack_time = timbre.get("attack", 0.01)
    attack_samples = max(int(attack_time * SAMPLE_RATE), 1)
    envelope = np.ones(n_active)

    if attack_samples < n_active:
        attack_curve = np.linspace(0, 1, min(attack_samples, n_active))
        attack_curve = 1.0 - np.exp(-3.5 * attack_curve)
        attack_curve /= max(attack_curve[-1], 1e-8)
        envelope[:len(attack_curve)] = attack_curve

    if timbre.get("decay_shape") == "exp":
        envelope *= np.exp(-2.5 * t_norm)
    else:
        envelope *= np.maximum(0, 1.0 - t_norm * 0.25)
        envelope *= np.exp(-0.8 * t_norm)

    fadeout = min(int(0.015 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        envelope[-fadeout:] *= np.linspace(1, 0, fadeout)

    # --- 6. ATTACK NOISE ---
    noise_amount = timbre.get("noise_amount", 0)
    if noise_amount > 0:
        noise = generate_noise(
            n_active, timbre.get("noise_type", "click"),
            timbre.get("noise_bandwidth", "wide"), rng
        )
        if timbre.get("noise_type") == "click":
            noise_env = np.exp(-np.linspace(0, 30, n_active))
        elif timbre.get("noise_type") == "bow":
            noise_env = 0.3 + 0.7 * np.exp(-np.linspace(0, 6, n_active))
            noise_env *= 1.0 + 0.15 * np.sin(
                2 * np.pi * rng.uniform(0.5, 2.0) * t_local
            )
        else:
            noise_env = 0.15 + 0.85 * np.exp(-np.linspace(0, 8, n_active))

        signal += noise * noise_env * noise_amount

    voice[mask] = signal * envelope * amplitude
    return voice
