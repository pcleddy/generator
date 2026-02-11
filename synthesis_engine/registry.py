"""
registry.py — InstrumentRegistry: single source of truth for all instrument profiles.

Loads all profiles at init. Eliminates duplication across composition files.
"""

from .profiles.strings import TIMBRES, STRING_PROFILES
from .profiles.bells import BELL_PROFILES
from .profiles.wood import WOOD_PROFILES
from .profiles.brass import BRASS_PROFILES
from .profiles.voices import VOICE_TIMBRES
from .profiles.plucked import PLUCKED_PROFILES
from .profiles.fm_brass import FM_BRASS_PROFILES


class InstrumentRegistry:
    """Centralized instrument profile database.

    Usage:
        reg = InstrumentRegistry()
        profile = reg.get("church_bell")
        all_bells = reg.list_category("bell")
        reg.register("my_custom_bell", {...})
    """

    def __init__(self):
        self._profiles = {}
        self._load_all()

    def _load_all(self):
        """Load all built-in profiles."""
        # String/wind timbres (from webern_pointillism)
        for name, profile in TIMBRES.items():
            self._profiles[name] = profile

        # Orchestral strings (from mahler_hail)
        for name, profile in STRING_PROFILES.items():
            self._profiles[name] = profile

        # Bells
        for name, profile in BELL_PROFILES.items():
            self._profiles[name] = profile

        # Wood percussion
        for name, profile in WOOD_PROFILES.items():
            self._profiles[name] = profile

        # Brass
        for name, profile in BRASS_PROFILES.items():
            self._profiles[name] = profile

        # Voices
        for name, profile in VOICE_TIMBRES.items():
            self._profiles[name] = profile

        # Plucked strings (Karplus-Strong)
        for name, profile in PLUCKED_PROFILES.items():
            self._profiles[name] = profile

        # FM Brass
        for name, profile in FM_BRASS_PROFILES.items():
            self._profiles[name] = profile

    def get(self, name):
        """Get profile by name. Raises KeyError if not found."""
        if name not in self._profiles:
            raise KeyError(
                f"Instrument '{name}' not found. "
                f"Available: {sorted(self._profiles.keys())}"
            )
        return self._profiles[name]

    def register(self, name, profile):
        """Register a new or custom profile."""
        if isinstance(profile, dict) and 'name' not in profile:
            profile['name'] = name
        self._profiles[name] = profile

    def has(self, name):
        """Check if a profile exists."""
        return name in self._profiles

    def list_all(self):
        """List all registered instrument names."""
        return sorted(self._profiles.keys())

    def list_category(self, category):
        """Get all instruments in a category (bell, string, voice, etc.)."""
        return sorted(
            name for name, p in self._profiles.items()
            if p.get('category', '') == category
        )

    def categories(self):
        """List all available categories."""
        return sorted(set(
            p.get('category', 'unknown')
            for p in self._profiles.values()
        ))

    def __len__(self):
        return len(self._profiles)

    def __contains__(self, name):
        return name in self._profiles

    def __repr__(self):
        cats = {}
        for p in self._profiles.values():
            cat = p.get('category', 'unknown')
            cats[cat] = cats.get(cat, 0) + 1
        cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cats.items()))
        return f"InstrumentRegistry({len(self)} instruments: {cat_str})"
