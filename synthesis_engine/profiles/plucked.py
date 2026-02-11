"""
plucked.py — Karplus-Strong plucked string instrument profiles.

Each profile tunes the K-S parameters for a specific instrument character.
"""

PLUCKED_PROFILES = {
    "guitar_nylon": {
        "name": "guitar_nylon",
        "category": "plucked",
        "loss_factor": 0.997,
        "brightness": 0.4,      # warm, round tone
        "pick_position": 0.5,   # middle of string
        "pluck_shape": "noise",
        "stiffness": 0.001,
        "body_resonance": 0.35,
        "body_freq": 280,       # classical guitar body
        "damping_speed": 1.0,
    },
    "guitar_steel": {
        "name": "guitar_steel",
        "category": "plucked",
        "loss_factor": 0.998,
        "brightness": 0.65,     # brighter than nylon
        "pick_position": 0.35,  # closer to bridge
        "pluck_shape": "noise",
        "stiffness": 0.003,
        "body_resonance": 0.3,
        "body_freq": 320,
        "damping_speed": 0.9,
    },
    "harp": {
        "name": "harp",
        "category": "plucked",
        "loss_factor": 0.996,
        "brightness": 0.55,
        "pick_position": 0.45,
        "pluck_shape": "triangle",  # smooth finger pluck
        "stiffness": 0.002,
        "body_resonance": 0.25,
        "body_freq": 200,       # large resonant body
        "damping_speed": 1.1,   # slightly faster decay
    },
    "plucked_cello": {
        "name": "plucked_cello",
        "category": "plucked",
        "loss_factor": 0.9985,  # long sustain
        "brightness": 0.35,     # dark, woody
        "pick_position": 0.55,
        "pluck_shape": "noise",
        "stiffness": 0.005,     # thick strings = more stiffness
        "body_resonance": 0.4,
        "body_freq": 250,       # cello body
        "damping_speed": 0.8,   # slow decay
    },
    "banjo": {
        "name": "banjo",
        "category": "plucked",
        "loss_factor": 0.993,   # fast decay — membrane head
        "brightness": 0.8,      # very bright, twangy
        "pick_position": 0.3,   # near bridge
        "pluck_shape": "sawtooth",  # sharp attack
        "stiffness": 0.004,
        "body_resonance": 0.5,  # strong membrane resonance
        "body_freq": 400,       # banjo head resonance is higher
        "damping_speed": 1.5,   # fast decay
    },
    "sitar": {
        "name": "sitar",
        "category": "plucked",
        "loss_factor": 0.998,
        "brightness": 0.7,
        "pick_position": 0.25,  # near bridge for buzz
        "pluck_shape": "sawtooth",
        "stiffness": 0.008,     # high stiffness for that sitar buzz
        "body_resonance": 0.2,
        "body_freq": 350,
        "damping_speed": 0.85,
    },
    "koto": {
        "name": "koto",
        "category": "plucked",
        "loss_factor": 0.996,
        "brightness": 0.6,
        "pick_position": 0.4,
        "pluck_shape": "noise",
        "stiffness": 0.002,
        "body_resonance": 0.3,
        "body_freq": 300,
        "damping_speed": 1.2,
    },
    "harpsichord": {
        "name": "harpsichord",
        "category": "plucked",
        "loss_factor": 0.994,   # short sustain — mechanical pluck
        "brightness": 0.85,     # very bright — quill pluck
        "pick_position": 0.2,   # very close to bridge
        "pluck_shape": "sawtooth",
        "stiffness": 0.006,
        "body_resonance": 0.15,
        "body_freq": 350,
        "damping_speed": 1.3,
    },
}
