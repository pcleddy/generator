"""
seed_manager.py — Centralized RNG for perfect reproducibility.

THE reproducibility guarantor. All composition code must use a SeedManager
instance, never raw random.Random() or np.random.RandomState() directly.

Same seed → same sequence of random calls → same events → same audio.
"""

import random
import numpy as np


class SeedManager:
    """Centralized RNG management.

    Wraps both Python's random and NumPy's RandomState so that a single
    seed controls all randomness in a composition. Call history can be
    logged for debugging/verification.

    Usage:
        rng = SeedManager(42)
        x = rng.uniform(0.0, 1.0)
        n = rng.randint(3, 8)
        arr = rng.randn(1024)
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.seed = seed
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._call_count = 0

    def set_seed(self, seed):
        """Reset both RNGs with a new seed."""
        self.seed = seed
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._call_count = 0

    # --- Scalar random values (Python random) ---

    def uniform(self, a, b):
        """Random float in [a, b]."""
        self._call_count += 1
        return self._py_rng.uniform(a, b)

    def randint(self, a, b):
        """Random integer in [a, b] inclusive."""
        self._call_count += 1
        return self._py_rng.randint(a, b)

    def random(self):
        """Random float in [0, 1)."""
        self._call_count += 1
        return self._py_rng.random()

    def choice(self, seq, weights=None):
        """Random choice from sequence."""
        self._call_count += 1
        if weights:
            return self._py_rng.choices(seq, weights=weights, k=1)[0]
        return self._py_rng.choice(seq)

    def choices(self, seq, weights=None, k=1):
        """Random choices with replacement."""
        self._call_count += 1
        return self._py_rng.choices(seq, weights=weights, k=k)

    def shuffle(self, seq):
        """In-place shuffle."""
        self._call_count += 1
        self._py_rng.shuffle(seq)

    def sample(self, seq, k):
        """Random sample without replacement."""
        self._call_count += 1
        return self._py_rng.sample(seq, k)

    def gauss(self, mu=0.0, sigma=1.0):
        """Gaussian random value."""
        self._call_count += 1
        return self._py_rng.gauss(mu, sigma)

    # --- NumPy array random values ---

    def randn(self, *shape):
        """NumPy standard normal array."""
        self._call_count += 1
        return self._np_rng.randn(*shape)

    def np_uniform(self, low=0.0, high=1.0, size=None):
        """NumPy uniform array."""
        self._call_count += 1
        return self._np_rng.uniform(low, high, size)

    def np_randint(self, low, high=None, size=None):
        """NumPy random integers."""
        self._call_count += 1
        return self._np_rng.randint(low, high, size)

    def np_choice(self, a, size=None, replace=True, p=None):
        """NumPy random choice."""
        self._call_count += 1
        return self._np_rng.choice(a, size=size, replace=replace, p=p)

    # --- Diagnostics ---

    @property
    def call_count(self):
        """Total RNG calls made (for reproducibility debugging)."""
        return self._call_count

    def __repr__(self):
        return f"SeedManager(seed={self.seed}, calls={self._call_count})"
