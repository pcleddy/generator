"""
strings.py — String and wind instrument timbre profiles.

Extracted from webern_pointillism.py TIMBRES list.
Each profile defines harmonics, envelope, noise, vibrato, and drift.
"""

# These are used by pointillist_tone() for the core synthesis engine.
# Keys match the original TIMBRES[n]["name"] values.

TIMBRES = {
    "cello_pont": {
        "name": "cello_pont",
        "category": "string",
        "harmonics": [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08],
        "partial_decay": 1.4,
        "attack": 0.06,
        "noise_type": "bow", "noise_amount": 0.12, "noise_bandwidth": 3000,
        "vibrato_delay": 0.3, "vibrato_rate": 4.5, "vibrato_depth": 12,
        "drift": 5, "undertone": 0.0, "decay_shape": "sustained",
    },
    "cello_tasto": {
        "name": "cello_tasto",
        "category": "string",
        "harmonics": [1.0, 0.3, 0.08, 0.02],
        "partial_decay": 2.0,
        "attack": 0.12,
        "noise_type": "bow", "noise_amount": 0.08, "noise_bandwidth": 1200,
        "vibrato_delay": 0.5, "vibrato_rate": 4.2, "vibrato_depth": 15,
        "drift": 8, "undertone": 0.03, "decay_shape": "sustained",
    },
    "flute_breathy": {
        "name": "flute_breathy",
        "category": "wind",
        "harmonics": [1.0, 0.12, 0.04],
        "partial_decay": 1.8,
        "attack": 0.04,
        "noise_type": "breath", "noise_amount": 0.18, "noise_bandwidth": 5000,
        "vibrato_delay": 0.2, "vibrato_rate": 5.0, "vibrato_depth": 8,
        "drift": 3, "undertone": 0.0, "decay_shape": "sustained",
    },
    "clarinet_chalumeau": {
        "name": "clarinet_chalumeau",
        "category": "wind",
        "harmonics": [1.0, 0.02, 0.6, 0.01, 0.3, 0.01, 0.1],  # odd partials
        "partial_decay": 1.3,
        "attack": 0.025,
        "noise_type": "breath", "noise_amount": 0.06, "noise_bandwidth": 2200,
        "vibrato_delay": 0.4, "vibrato_rate": 4.8, "vibrato_depth": 6,
        "drift": 4, "undertone": 0.0, "decay_shape": "sustained",
    },
    "bell_struck": {
        "name": "bell_struck",
        "category": "pitched_percussion",
        "harmonics": [1.0, 0.65, 0.4, 0.2, 0.12, 0.06, 0.03, 0.015],
        "partial_decay": 0.8,
        "attack": 0.002,
        "noise_type": "click", "noise_amount": 0.15, "noise_bandwidth": "wide",
        "vibrato_delay": 99, "vibrato_rate": 0, "vibrato_depth": 0,
        "drift": 1, "undertone": 0.04, "decay_shape": "exp",
    },
    "glass_harmonica": {
        "name": "glass_harmonica",
        "category": "pitched_percussion",
        "harmonics": [1.0, 0.0, 0.0, 0.35, 0.0, 0.18],
        "partial_decay": 0.9,
        "attack": 0.08,
        "noise_type": "breath", "noise_amount": 0.04, "noise_bandwidth": 6000,
        "vibrato_delay": 0.1, "vibrato_rate": 5.5, "vibrato_depth": 4,
        "drift": 2, "undertone": 0.02, "decay_shape": "exp",
    },
    "pizzicato": {
        "name": "pizzicato",
        "category": "string",
        "harmonics": [1.0, 0.55, 0.3, 0.18, 0.1, 0.05],
        "partial_decay": 1.6,
        "attack": 0.001,
        "noise_type": "click", "noise_amount": 0.2, "noise_bandwidth": "wide",
        "vibrato_delay": 99, "vibrato_rate": 0, "vibrato_depth": 0,
        "drift": 6, "undertone": 0.0, "decay_shape": "exp",
    },
    "oboe_pp": {
        "name": "oboe_pp",
        "category": "wind",
        "harmonics": [1.0, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04],
        "partial_decay": 1.2,
        "attack": 0.015,
        "noise_type": "breath", "noise_amount": 0.05, "noise_bandwidth": 3500,
        "vibrato_delay": 0.25, "vibrato_rate": 5.2, "vibrato_depth": 10,
        "drift": 4, "undertone": 0.0, "decay_shape": "sustained",
    },
}

# Mahler orchestral strings (from mahler_hail.py)
STRING_PROFILES = {
    "violin": {
        "name": "violin",
        "category": "string",
        "harmonics": [1.0, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08, 0.05],
        "attack": 0.04,
        "noise_type": "bow", "noise_amount": 0.06,
        "vibrato_rate": 5.5, "vibrato_depth": 12, "vibrato_delay": 0.15,
    },
    "viola": {
        "name": "viola",
        "category": "string",
        "harmonics": [1.0, 0.8, 0.55, 0.4, 0.25, 0.15, 0.08],
        "attack": 0.05,
        "noise_type": "bow", "noise_amount": 0.07,
        "vibrato_rate": 5.0, "vibrato_depth": 14, "vibrato_delay": 0.18,
    },
    "cello": {
        "name": "cello",
        "category": "string",
        "harmonics": [1.0, 0.85, 0.6, 0.45, 0.3, 0.18, 0.1, 0.06],
        "attack": 0.06,
        "noise_type": "bow", "noise_amount": 0.08,
        "vibrato_rate": 4.5, "vibrato_depth": 16, "vibrato_delay": 0.2,
    },
    "bass": {
        "name": "bass",
        "category": "string",
        "harmonics": [1.0, 0.9, 0.65, 0.5, 0.35, 0.2, 0.12],
        "attack": 0.08,
        "noise_type": "bow", "noise_amount": 0.09,
        "vibrato_rate": 4.0, "vibrato_depth": 10, "vibrato_delay": 0.25,
    },
}
