"""
voices.py — Vocal synthesis profiles.

Extracted from berg_vocal.py.
Vowel formant data, voice types, formant shift multipliers.
"""

VOWELS = {
    "ah": [
        (800,  80,  1.0),
        (1150, 90,  0.63),
        (2800, 120, 0.15),
        (3500, 130, 0.07),
        (4950, 140, 0.03),
    ],
    "ee": [
        (270,  60,  1.0),
        (2300, 90,  0.50),
        (3000, 110, 0.12),
        (3700, 130, 0.05),
        (4950, 140, 0.02),
    ],
    "oh": [
        (500,  70,  1.0),
        (700,  80,  0.55),
        (2800, 100, 0.10),
        (3500, 130, 0.05),
        (4950, 140, 0.02),
    ],
    "oo": [
        (300,  50,  1.0),
        (600,  70,  0.45),
        (2300, 100, 0.06),
        (3500, 120, 0.03),
        (4950, 140, 0.01),
    ],
    "mm": [
        (250,  50,  1.0),
        (1700, 100, 0.15),
        (2500, 120, 0.05),
        (3300, 140, 0.02),
    ],
    "eh": [
        (530,  60,  1.0),
        (1850, 90,  0.50),
        (2500, 110, 0.10),
        (3500, 130, 0.05),
        (4950, 140, 0.02),
    ],
}

VOICE_FORMANT_SHIFT = {
    "soprano": 1.15,
    "alto":    1.0,
    "tenor":   0.88,
    "bass":    0.78,
}

VOICE_TIMBRES = {
    "soprano": {
        "name": "soprano",
        "category": "voice",
        "octave_range": (4, 5),
        "vowel_sequence": ["mm", "ah", "ee", "ah", "mm"],
        "breathiness": 0.07,
        "jitter": 0.008,
        "shimmer": 0.06,
        "vibrato_rate": 5.8,
        "vibrato_depth": 18,
        "vibrato_delay": 0.15,
        "drift": 4,
        "attack": 0.08,
        "formant_shift": "soprano",
    },
    "alto": {
        "name": "alto",
        "category": "voice",
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
    "tenor": {
        "name": "tenor",
        "category": "voice",
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
    "bass": {
        "name": "bass",
        "category": "voice",
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
}

BODY_RESONANCES = {
    "cello":     [(280, 50), (500, 60), (700, 70), (1200, 100)],
    "clarinet":  [(350, 40), (900, 80), (1500, 100), (2800, 120)],
    "flute":     [(800, 100), (1600, 120), (3200, 140)],
    "oboe":      [(1000, 60), (1800, 80), (2800, 100), (3500, 120)],
    "bell":      [(600, 40), (1200, 60), (2400, 80), (4800, 100)],
    "glass":     [(500, 50), (1500, 80), (3000, 100)],
    "pizz":      [(300, 60), (700, 80), (1400, 100)],
}
