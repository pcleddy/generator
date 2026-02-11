"""
composition.py — Composition base class.

All pieces subclass Composition. Events are generated deterministically
from seed + parameters, then rendered to audio by the Renderer.
"""

import json

from .seed_manager import SeedManager
from .registry import InstrumentRegistry
from .event import SynthEvent
from .config import pitch_to_name


class Composition:
    """Base class for all compositions.

    Subclasses implement generate() which schedules events via add_event().
    Same seed + same parameters = identical events, always.

    Usage:
        class MyPiece(Composition):
            def generate(self):
                self.add_event(time=0.5, pc=2, octave=4, duration=2.0,
                              amplitude=0.15, instrument='church_bell',
                              category='melody')

        piece = MyPiece(seed=42, duration=30)
        events = piece.generate()
        piece.save_json('my_piece.json')
    """

    def __init__(self, seed=42, duration=60.0, registry=None,
                 title="Untitled", reverb_preset="room", **params):
        self.seed = seed
        self.duration = duration
        self.title = title
        self.reverb_preset = reverb_preset
        self.params = params

        self.registry = registry or InstrumentRegistry()
        self.rng = SeedManager(seed)
        self.events = []

    def generate(self):
        """Generate all events. Subclasses must implement this.

        Must be idempotent: same seed → same events every time.
        Returns: list of SynthEvent
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def add_event(self, time, pc, octave, duration, amplitude, instrument,
                  category="", section="", **extra_params):
        """Schedule a synthesis event. Auto-logged, auto-numbered."""
        event = SynthEvent(
            time=float(time),
            pitch_class=int(pc),
            octave=int(octave),
            duration=float(duration),
            amplitude=float(amplitude),
            instrument=str(instrument),
            category=str(category),
            section=str(section),
            parameters=extra_params,
        )
        self.events.append(event)
        return event

    def sorted_events(self):
        """Return events sorted by time."""
        return sorted(self.events, key=lambda e: e.time)

    def event_count(self):
        """Total number of events."""
        return len(self.events)

    def events_by_category(self):
        """Group events by category. Returns dict of category → count."""
        counts = {}
        for e in self.events:
            cat = e.category or e.instrument
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def to_dict(self):
        """Serialize composition metadata + all events to dict."""
        return {
            'title': self.title,
            'duration': self.duration,
            'seed': self.seed,
            'audio_file': '',  # filled by renderer
            'reverb_preset': self.reverb_preset,
            'params': self.params,
            'events': [e.to_dict() for e in self.sorted_events()],
        }

    def save_json(self, path, audio_file=None):
        """Export events as JSON for the animated score player."""
        data = self.to_dict()
        if audio_file:
            data['audio_file'] = audio_file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    def summary(self):
        """Print a summary of the composition."""
        cats = self.events_by_category()
        lines = [
            f"{self.title}",
            f"  Seed: {self.seed}  Duration: {self.duration}s  Events: {len(self.events)}",
            f"  Reverb: {self.reverb_preset}",
            f"  RNG calls: {self.rng.call_count}",
        ]
        if cats:
            lines.append("  Categories:")
            for cat, count in sorted(cats.items()):
                lines.append(f"    {cat}: {count}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"{self.__class__.__name__}(seed={self.seed}, "
                f"duration={self.duration}, events={len(self.events)})")
