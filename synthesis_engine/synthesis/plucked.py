"""
plucked.py — Karplus-Strong plucked string synthesis.

Physical modeling: a short noise burst feeds into a feedback delay line
with a lowpass filter. The delay line length sets the pitch; the filter
controls decay rate and brightness evolution.

Real strings: higher harmonics die faster (the lowpass does this naturally),
the initial pluck shape affects attack character, and stiffness causes
slight inharmonicity in the upper partials.

Extensions beyond basic K-S:
  - Fractional delay (allpass interpolation) for accurate pitch tuning
  - Pick position filtering (comb filter notches)
  - Pluck shape control (noise vs shaped excitation)
  - Loss factor for decay rate control
  - Stiffness filter for piano-like inharmonicity
"""

import numpy as np
from ..config import SAMPLE_RATE


def karplus_strong(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a plucked string using extended Karplus-Strong.

    Args:
        t: global time array
        start: note onset time (seconds)
        freq: fundamental frequency (Hz)
        duration: note duration (seconds)
        amplitude: 0-1 amplitude
        profile: dict with K-S parameters (see PLUCKED_PROFILES)
        rng: SeedManager instance

    Returns:
        numpy array same length as t, with the plucked tone mixed in
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)

    if freq < 20 or freq > 8000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = int(np.sum(mask))
    if n_active == 0:
        return voice

    # ── Delay line length ──
    # Exact period in samples (fractional)
    period_exact = SAMPLE_RATE / freq
    delay_len = int(np.floor(period_exact))
    fractional = period_exact - delay_len

    if delay_len < 2:
        return voice

    # ── Allpass coefficient for fractional delay tuning ──
    # This corrects the pitch to sub-cent accuracy
    if fractional > 0.001:
        C = (1.0 - fractional) / (1.0 + fractional)
    else:
        C = 0.0

    # ── Profile parameters ──
    loss = profile.get("loss_factor", 0.998)
    brightness = profile.get("brightness", 0.5)  # 0=dark, 1=bright
    pick_pos = profile.get("pick_position", 0.5)  # 0-1, position along string
    pluck_shape = profile.get("pluck_shape", "noise")  # noise, sine, sawtooth
    stiffness = profile.get("stiffness", 0.0)  # 0-0.05, inharmonicity
    body_resonance = profile.get("body_resonance", 0.0)  # 0-1, body coupling
    damping_speed = profile.get("damping_speed", 1.0)  # multiplier on decay

    # Adjust loss for requested damping speed
    effective_loss = loss ** damping_speed

    # ── Initial excitation ──
    excitation = np.zeros(delay_len)
    if pluck_shape == "noise":
        excitation = rng.randn(delay_len) * 0.5
    elif pluck_shape == "sine":
        # Single cycle of sine — smoother pluck
        excitation = np.sin(2 * np.pi * np.arange(delay_len) / delay_len)
    elif pluck_shape == "sawtooth":
        # Bright, sharp pluck
        excitation = np.linspace(1, -1, delay_len)
    elif pluck_shape == "triangle":
        # Warm pluck — triangle wave
        half = delay_len // 2
        excitation[:half] = np.linspace(0, 1, half)
        excitation[half:] = np.linspace(1, -(delay_len - half - 1) / max(half, 1), delay_len - half)

    # ── Pick position filter ──
    # Comb filter that notches harmonics near the pick point
    # A pick at position p notches the (1/p)th harmonic and multiples
    if pick_pos > 0.01 and pick_pos < 0.99:
        pick_delay = max(int(delay_len * pick_pos), 1)
        if pick_delay < len(excitation):
            # Subtract a delayed copy — creates the comb notch
            shifted = np.zeros_like(excitation)
            shifted[pick_delay:] = excitation[:-pick_delay]
            excitation = excitation - 0.5 * shifted

    # Normalize excitation
    ex_peak = np.max(np.abs(excitation))
    if ex_peak > 0:
        excitation /= ex_peak

    # ── Brightness filter coefficients ──
    # Controls the lowpass in the feedback loop
    # brightness=1 → minimal filtering (bright), brightness=0 → heavy filtering (dark)
    # Two-point average with blend: y = b*x[n] + (1-b)*x[n-1]
    b_bright = 0.3 + 0.5 * brightness  # range 0.3 (dark) to 0.8 (bright)

    # ── Karplus-Strong synthesis loop ──
    # This is the core — can't be vectorized due to feedback
    output = np.zeros(n_active)
    buffer = excitation.copy()
    allpass_prev_in = 0.0
    allpass_prev_out = 0.0

    for i in range(n_active):
        # Read from delay line
        idx0 = i % delay_len
        idx1 = (i + 1) % delay_len

        # Two-point averaging lowpass filter
        sample = b_bright * buffer[idx0] + (1.0 - b_bright) * buffer[idx1]

        # Apply loss factor
        sample *= effective_loss

        # Allpass filter for fractional delay (pitch accuracy)
        if C != 0.0:
            allpass_out = C * sample + allpass_prev_in - C * allpass_prev_out
            allpass_prev_in = sample
            allpass_prev_out = allpass_out
            sample = allpass_out

        # Stiffness: slight pitch sharpening of upper partials
        # (allpass dispersion — approximated by mixing in a slightly
        # different delay tap)
        if stiffness > 0.001:
            idx_stiff = (i + 2) % delay_len
            sample = (1.0 - stiffness) * sample + stiffness * buffer[idx_stiff]

        # Write back to buffer (feedback)
        buffer[idx0] = sample
        output[i] = sample

    # ── Body resonance ──
    # Simulate a resonant body (guitar body, cello body, etc.)
    # Simple: add a low-frequency emphasis via 2nd-order resonance
    if body_resonance > 0.01:
        from scipy.signal import lfilter
        # Body resonance around 200-400 Hz (guitar-ish)
        body_freq = profile.get("body_freq", 280)
        w0 = 2 * np.pi * body_freq / SAMPLE_RATE
        Q = 3.0
        alpha = np.sin(w0) / (2 * Q)
        b_filt = [alpha, 0, -alpha]
        a_filt = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
        body_signal = lfilter(b_filt, a_filt, output)
        bp = np.max(np.abs(body_signal))
        if bp > 0:
            body_signal /= bp
        output = output + body_resonance * body_signal

    # ── Envelope shaping ──
    # K-S naturally decays, but add a soft onset and fadeout
    attack_len = max(int(0.002 * SAMPLE_RATE), 1)  # 2ms click avoidance
    if attack_len < n_active:
        output[:attack_len] *= np.linspace(0, 1, attack_len)

    fadeout = min(int(0.01 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        output[-fadeout:] *= np.linspace(1, 0, fadeout)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak

    voice[mask] = output * amplitude
    return voice
