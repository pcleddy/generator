"""
percussion.py — Non-pitched and specialized percussion profiles.

Cowbell, timpani, hail pellet partial ratios.
Extracted from mahler_hail.py.
"""

# Cowbell: distinctly inharmonic metallic partials
COWBELL_PARTIALS = [
    (1.0,   1.0,  3.0),
    (1.504, 0.7,  4.0),
    (1.836, 0.5,  3.5),
    (2.0,   0.3,  5.0),
    (2.536, 0.4,  4.5),
    (3.17,  0.2,  6.0),
]

# Timpani: circular membrane modes (inharmonic)
TIMPANI_MODES = [
    (1.0,   1.0,  2.5),   # fundamental
    (1.504, 0.6,  3.5),   # (1,1) mode
    (1.742, 0.3,  4.0),   # (2,1) mode
    (2.0,   0.5,  3.0),   # (0,2) mode
    (2.296, 0.2,  5.0),   # (3,1) mode
]

# Prepared piano preparation types
PREPARATIONS = ["bolt", "rubber", "screw"]
