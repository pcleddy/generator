"""
wood.py — Wood percussion profiles: xylophone, marimba.

Extracted from bells_gentle.py.
"""

WOOD_PROFILES = {
    "wood_xylophone": {
        "name": "wood_xylophone",
        "category": "wood",
        "partials": [
            (1.0,   1.0,   3.0),
            (3.0,   0.15,  5.0),
            (4.0,   0.40,  4.0),   # tuned octave-double — the bright "wood" sound
            (6.27,  0.06,  7.0),
        ],
        "strike_brightness": 4000,
        "strike_amount": 0.10,
        "ring_time_mult": 0.6,
    },
    "marimba": {
        "name": "marimba",
        "category": "wood",
        "partials": [
            (1.0,   1.0,   2.0),
            (4.0,   0.25,  3.5),
            (2.8,   0.05,  5.0),
        ],
        "strike_brightness": 2000,
        "strike_amount": 0.06,
        "ring_time_mult": 1.0,
    },
}
