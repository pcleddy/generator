"""
reverb.py — Multi-tap delay reverb with named presets.

Extracted from webern_pointillism.py simple_reverb() with presets
consolidated from all composition files.
"""

import numpy as np

# Named reverb presets (delay times in ms, all prime-ish for diffusion)
REVERB_PRESETS = {
    "intimate":      {"delays_ms": [13, 19, 29, 41, 59],                          "decay": 0.25, "wet": 0.12},
    "room":          {"delays_ms": [17, 29, 43, 61, 79],                           "decay": 0.30, "wet": 0.15},
    "chamber":       {"delays_ms": [23, 37, 53, 71, 97, 131, 173, 229],           "decay": 0.35, "wet": 0.18},
    "box":           {"delays_ms": [7, 11, 17, 23, 31, 41, 53, 67],               "decay": 0.35, "wet": 0.12},
    "cathedral":     {"delays_ms": [31, 47, 67, 89, 113, 149, 191, 251, 313, 397],"decay": 0.55, "wet": 0.22},
    "concert_hall":  {"delays_ms": [29, 43, 61, 83, 109, 139, 179, 227, 283, 353, 431], "decay": 0.45, "wet": 0.22},
    "deep":          {"delays_ms": [37, 53, 71, 97, 131, 173, 229, 307, 401, 503],"decay": 0.50, "wet": 0.20},
}


def simple_reverb(audio, decay=0.4, delays_ms=None, sample_rate=44100,
                  wet_mix=None, preset=None):
    """Simple multi-tap delay reverb.

    Multiple feedback delay lines at prime-number-ratio intervals create
    a reasonable sense of space.

    Args:
        audio: input signal (numpy array)
        decay: overall decay amount (0-1)
        delays_ms: list of delay times in milliseconds
        sample_rate: audio sample rate
        wet_mix: wet signal mix amount (0-1). If None, uses preset or 0.18.
        preset: name of reverb preset (overrides delays_ms, decay, wet_mix)

    Returns:
        Processed audio with reverb applied.
    """
    if preset is not None:
        if preset not in REVERB_PRESETS:
            raise ValueError(
                f"Unknown reverb preset '{preset}'. "
                f"Available: {list(REVERB_PRESETS.keys())}"
            )
        p = REVERB_PRESETS[preset]
        delays_ms = p["delays_ms"]
        decay = p["decay"]
        if wet_mix is None:
            wet_mix = p["wet"]

    if delays_ms is None:
        delays_ms = [23, 37, 53, 71, 97, 131, 173, 229]

    if wet_mix is None:
        wet_mix = 0.18

    dry_mix = 1.0 - wet_mix

    wet = np.zeros_like(audio)

    for i, delay_ms in enumerate(delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000)
        tap_gain = decay * (0.85 ** i)

        delayed = np.zeros_like(audio)
        if delay_samples < len(audio):
            delayed[delay_samples:] = audio[:-delay_samples] * tap_gain

        wet += delayed

    return audio * dry_mix + wet * wet_mix
