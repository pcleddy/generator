"""
bells.py — Bell profiles: glockenspiel, celesta, tubular, church, wind chime, papa, music box.

Extracted from bells_bergman.py, cage_bells_family.py, quay_music_box.py.
Real bells have specific partial relationships (not harmonic!):
  Hum: ~0.5×, Prime: 1.0×, Tierce: ~1.183×, Quint: ~1.506×, Nominal: ~2.0×
"""

BELL_PROFILES = {
    "glockenspiel": {
        "name": "glockenspiel",
        "category": "bell",
        "partials": [
            (1.0,   1.0,   3.5),
            (2.76,  0.45,  4.0),
            (5.40,  0.25,  5.5),
            (8.93,  0.12,  7.0),
            (13.3,  0.06,  9.0),
        ],
        "strike_brightness": 8000,
        "strike_amount": 0.20,
        "ring_time_mult": 0.7,
    },
    "celesta": {
        "name": "celesta",
        "category": "bell",
        "partials": [
            (1.0,   1.0,   2.8),
            (2.0,   0.35,  3.5),
            (3.0,   0.15,  4.5),
            (5.2,   0.08,  6.0),
        ],
        "strike_brightness": 5000,
        "strike_amount": 0.12,
        "ring_time_mult": 1.0,
    },
    "tubular_bell": {
        "name": "tubular_bell",
        "category": "bell",
        "partials": [
            (0.5,   0.25,  1.5),
            (1.0,   1.0,   1.8),
            (1.183, 0.70,  2.0),
            (1.506, 0.45,  2.3),
            (2.0,   0.55,  2.5),
            (2.514, 0.20,  3.0),
            (3.011, 0.12,  3.5),
            (4.166, 0.05,  5.0),
        ],
        "strike_brightness": 3000,
        "strike_amount": 0.15,
        "ring_time_mult": 2.0,
    },
    "church_bell": {
        "name": "church_bell",
        "category": "bell",
        "partials": [
            (0.5,   0.35,  1.0),
            (1.0,   1.0,   1.2),
            (1.183, 0.80,  1.3),
            (1.506, 0.55,  1.5),
            (2.0,   0.65,  1.6),
            (2.514, 0.30,  2.0),
            (2.662, 0.22,  2.2),
            (3.011, 0.15,  2.5),
            (4.166, 0.08,  3.5),
            (5.433, 0.04,  4.5),
        ],
        "strike_brightness": 2000,
        "strike_amount": 0.18,
        "ring_time_mult": 3.0,
    },
    "wind_chime": {
        "name": "wind_chime",
        "category": "bell",
        "partials": [
            (1.0,   1.0,   4.0),
            (2.756, 0.50,  4.5),
            (5.404, 0.30,  5.0),
            (8.933, 0.15,  6.0),
            (13.34, 0.08,  8.0),
        ],
        "strike_brightness": 9000,
        "strike_amount": 0.08,
        "ring_time_mult": 0.5,
    },
    # Papa's custom 10-partial bright profile — for audibility in low registers
    "papa_bell": {
        "name": "papa_bell",
        "category": "bell",
        "partials": [
            (0.5,   0.30,  1.2),
            (1.0,   1.0,   1.5),
            (1.183, 0.70,  1.8),
            (1.506, 0.50,  2.0),
            (2.0,   0.65,  2.0),
            (2.514, 0.40,  2.5),
            (2.662, 0.35,  2.8),
            (3.011, 0.30,  3.0),
            (4.0,   0.20,  3.5),
            (5.0,   0.12,  4.0),
        ],
        "strike_brightness": 3500,
        "strike_amount": 0.12,
        "ring_time_mult": 2.5,
    },
    # Music box tine
    "music_box_tine": {
        "name": "music_box_tine",
        "category": "bell",
        "partials": [
            (1.0,   1.0,   4.0),
            (2.0,   0.55,  3.5),
            (3.0,   0.25,  5.0),
            (4.0,   0.12,  6.0),
            (5.404, 0.35,  2.5),  # THE music box metallic overtone
            (6.0,   0.08,  7.0),
        ],
        "strike_brightness": 8000,
        "strike_amount": 0.15,
        "ring_time_mult": 0.8,
    },
    # Worn music box tine (slightly detuned)
    "music_box_worn": {
        "name": "music_box_worn",
        "category": "bell",
        "partials": [
            (1.0,   1.0,   4.5),
            (1.997, 0.50,  4.0),
            (3.01,  0.22,  5.5),
            (4.0,   0.10,  6.5),
            (5.38,  0.30,  3.0),
            (6.1,   0.06,  8.0),
        ],
        "strike_brightness": 7000,
        "strike_amount": 0.12,
        "ring_time_mult": 0.7,
    },
}
