"""
rows.py — 12-tone row operations and pre-built rows.

Consolidated from webern_pointillism.py, berg_lyrical.py, berg_extended.py, berg_vocal.py.
One definition of each transform. No more duplication.
"""

import numpy as np


# =====================================================================
# Pre-built rows
# =====================================================================

BERG_ROW = np.array([7, 10, 2, 6, 9, 0, 4, 8, 11, 1, 3, 5])
WEBERN_TRICHORD = [0, 1, 4]


# =====================================================================
# Row operations
# =====================================================================

def derive_row(trichord):
    """Derive a 12-tone row from a trichord using Webern's method.

    Build 4 forms of the trichord (original, inversion, retrograde,
    retrograde-inversion), transpose each to start on the next available
    pitch class, concatenate.
    """
    trichord = np.array(trichord)
    row = list(trichord)
    intervals = np.diff(trichord) % 12

    # Inversion: negate intervals
    inv_intervals = (-intervals) % 12
    inv_start = (trichord[-1] + intervals[0]) % 12
    inv = [inv_start]
    for itv in inv_intervals:
        inv.append((inv[-1] + itv) % 12)

    # Retrograde
    retro_start = (inv[-1] + intervals[0]) % 12
    retro_intervals = intervals[::-1]
    retro = [retro_start]
    for itv in retro_intervals:
        retro.append((retro[-1] + itv) % 12)

    # Retrograde-inversion
    ri_start = (retro[-1] + intervals[0]) % 12
    ri_intervals = inv_intervals[::-1]
    ri = [ri_start]
    for itv in ri_intervals:
        ri.append((ri[-1] + itv) % 12)

    row = list(trichord) + inv + retro + ri
    return np.array(row[:12])


def inversion(row):
    """Invert a row: mirror intervals around first note."""
    row = np.array(row)
    return (2 * row[0] - row) % 12


def retrograde(row):
    """Reverse a row."""
    return np.array(row)[::-1]


def transpose(row, interval):
    """Transpose a row by a given interval (semitones)."""
    return (np.array(row) + interval) % 12


def retrograde_inversion(row):
    """Combine retrograde and inversion."""
    return retrograde(inversion(row))
