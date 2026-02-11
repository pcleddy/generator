"""
fm_brass.py — FM synthesis brass instrument profiles.

Based on DX7-era FM synthesis research. The key insight for brass:
brightness tracks loudness. The modulation index (which controls spectral
richness) should peak at the attack and decay to a lower sustain value.

All brass use carrier:modulator ratio 1:1 (harmonic spectrum).
The differences come from:
  - peak_mi / sustain_mi ratio (attack brightness)
  - envelope times (trumpet is snappy, tuba is slow)
  - feedback amount (adds warmth/complexity)
  - second operator pair mix (ensemble richness)
"""

FM_BRASS_PROFILES = {
    "fm_trumpet": {
        "name": "fm_trumpet",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 11.0,       # very bright attack
        "sustain_mi": 3.5,
        "mi_attack": 0.06,     # fast — trumpet snaps
        "mi_decay": 0.35,
        "amp_attack": 0.04,
        "amp_decay": 0.25,
        "amp_sustain": 0.75,
        "feedback": 0.5,       # minimal feedback
        "noise_amount": 0.08,
        "noise_decay": 0.25,
        "vibrato_rate": 5.5,
        "vibrato_depth": 6,
        "vibrato_delay": 0.3,
        "second_op_detune": 0.002,
        "second_op_mix": 0.25,
    },
    "fm_horn": {
        "name": "fm_horn",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 8.5,        # less bright — horn is warmer
        "sustain_mi": 3.0,
        "mi_attack": 0.12,     # slower attack — horn blooms
        "mi_decay": 0.5,
        "amp_attack": 0.08,
        "amp_decay": 0.35,
        "amp_sustain": 0.7,
        "feedback": 1.5,       # more feedback for warmth
        "noise_amount": 0.06,
        "noise_decay": 0.3,
        "vibrato_rate": 4.8,
        "vibrato_depth": 8,
        "vibrato_delay": 0.35,
        "second_op_detune": 0.003,
        "second_op_mix": 0.35,
    },
    "fm_trombone": {
        "name": "fm_trombone",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 13.0,       # bright blat on attack
        "sustain_mi": 4.5,
        "mi_attack": 0.08,
        "mi_decay": 0.45,
        "amp_attack": 0.06,
        "amp_decay": 0.3,
        "amp_sustain": 0.72,
        "feedback": 2.0,       # more body
        "noise_amount": 0.10,
        "noise_decay": 0.35,
        "vibrato_rate": 4.5,
        "vibrato_depth": 10,
        "vibrato_delay": 0.3,
        "second_op_detune": 0.0025,
        "second_op_mix": 0.3,
    },
    "fm_tuba": {
        "name": "fm_tuba",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 7.0,        # less bright — tuba is round
        "sustain_mi": 2.5,
        "mi_attack": 0.2,      # slow attack — big instrument
        "mi_decay": 0.7,
        "amp_attack": 0.15,
        "amp_decay": 0.4,
        "amp_sustain": 0.65,
        "feedback": 3.0,       # significant feedback for richness
        "noise_amount": 0.12,
        "noise_decay": 0.4,
        "vibrato_rate": 4.0,
        "vibrato_depth": 6,
        "vibrato_delay": 0.4,
        "second_op_detune": 0.004,
        "second_op_mix": 0.4,
    },
    "fm_flugelhorn": {
        "name": "fm_flugelhorn",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 6.5,        # dark, mellow
        "sustain_mi": 2.0,
        "mi_attack": 0.1,
        "mi_decay": 0.4,
        "amp_attack": 0.07,
        "amp_decay": 0.3,
        "amp_sustain": 0.7,
        "feedback": 1.0,
        "noise_amount": 0.05,
        "noise_decay": 0.2,
        "vibrato_rate": 5.0,
        "vibrato_depth": 8,
        "vibrato_delay": 0.25,
        "second_op_detune": 0.003,
        "second_op_mix": 0.35,
    },
    "fm_muted_trumpet": {
        "name": "fm_muted_trumpet",
        "category": "fm_brass",
        "mod_ratio": 1.0,
        "peak_mi": 14.0,       # very bright but filtered
        "sustain_mi": 5.0,     # stays edgier
        "mi_attack": 0.04,     # very snappy
        "mi_decay": 0.2,       # fast decay
        "amp_attack": 0.03,
        "amp_decay": 0.2,
        "amp_sustain": 0.6,
        "feedback": 2.5,       # nasal quality
        "noise_amount": 0.15,  # more breath noise (effort)
        "noise_decay": 0.15,
        "vibrato_rate": 6.0,
        "vibrato_depth": 10,
        "vibrato_delay": 0.2,
        "second_op_detune": 0.005,  # wider detune — mute resonance
        "second_op_mix": 0.2,
    },
}
