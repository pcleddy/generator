"""
envelopes.py — Shared envelope and noise generation functions.

Extracted from webern_pointillism.py.
"""

import numpy as np
from scipy.signal import lfilter

from ..config import SAMPLE_RATE


def generate_noise(n_samples, noise_type, bandwidth, rng):
    """Generate shaped noise for attack transients.

    Real instruments produce noise at the onset:
      - bow:    broadband friction filtered around string resonance
      - breath: turbulent airflow, bandpass around embouchure
      - click:  very short broadband impulse (pluck, key, hammer)
    """
    raw = rng.randn(n_samples)

    if noise_type == "click":
        click_len = int(0.002 * SAMPLE_RATE)
        envelope = np.zeros(n_samples)
        envelope[:min(click_len, n_samples)] = np.exp(
            -np.linspace(0, 8, min(click_len, n_samples))
        )
        return raw * envelope

    if bandwidth == "wide":
        return raw

    # Bandpass filter the noise around the characteristic frequency
    center = bandwidth
    width = center * 0.6
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = center / max(width, 1)
    alpha = np.sin(w0) / (2 * Q)

    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]

    filtered = lfilter(b, a, raw)
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak
    return filtered
