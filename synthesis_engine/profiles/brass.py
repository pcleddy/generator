"""
brass.py — Brass instrument profiles.

Extracted from mahler_hail.py.
"""

BRASS_PROFILES = {
    "horn": {
        "name": "horn",
        "category": "brass",
        "harmonics": [1.0, 0.9, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08, 0.05],
        "attack": 0.03,
        "noise_type": "breath", "noise_amount": 0.04,
        "vibrato_rate": 4.5, "vibrato_depth": 8, "vibrato_delay": 0.3,
        "brightness_boost": 1.0,
    },
    "trumpet": {
        "name": "trumpet",
        "category": "brass",
        "harmonics": [1.0, 0.95, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12, 0.07],
        "attack": 0.015,
        "noise_type": "breath", "noise_amount": 0.05,
        "vibrato_rate": 5.0, "vibrato_depth": 6, "vibrato_delay": 0.2,
        "brightness_boost": 1.3,
    },
    "trombone": {
        "name": "trombone",
        "category": "brass",
        "harmonics": [1.0, 0.85, 0.7, 0.6, 0.45, 0.3, 0.2, 0.12, 0.07],
        "attack": 0.025,
        "noise_type": "breath", "noise_amount": 0.06,
        "vibrato_rate": 4.0, "vibrato_depth": 10, "vibrato_delay": 0.25,
        "brightness_boost": 1.1,
    },
}
