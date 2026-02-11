"""
scales.py — All scale/mode definitions in one place.

Consolidated from bells_bergman.py, bells_pizz.py, bells_gentle.py, tubular_low.py.
"""

SCALES = {
    # Diatonic modes
    "D_DORIAN":      [2, 4, 5, 7, 9, 11, 0],
    "A_MIXOLYDIAN":  [9, 11, 1, 2, 4, 6, 7],
    "G_MAJOR":       [7, 9, 11, 0, 2, 4, 6],
    "G_LYDIAN":      [7, 9, 11, 1, 2, 4, 5],
    "D_MINOR":       [2, 4, 5, 7, 9, 10, 0],
    "E_PHRYGIAN":    [4, 5, 7, 9, 11, 0, 2],
    "F_MAJOR":       [5, 7, 9, 10, 0, 2, 4],

    # Pentatonic
    "A_PENT":        [9, 0, 2, 4, 7],
    "D_MIN_PENT":    [2, 5, 7, 9, 0],

    # Chromatic
    "CHROMATIC":     list(range(12)),
}


def get_scale(name):
    """Get scale by name. Case-insensitive, underscore-flexible."""
    key = name.upper().replace(' ', '_')
    if key in SCALES:
        return SCALES[key]
    raise KeyError(f"Scale '{name}' not found. Available: {list(SCALES.keys())}")
