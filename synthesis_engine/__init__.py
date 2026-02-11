"""
synthesis_engine — OOP-based algorithmic music synthesis.

No samples. No DAW. No MIDI. Just math.

Usage:
    from synthesis_engine import Composition, Renderer, SeedManager

    class MyPiece(Composition):
        def generate(self):
            self.add_event(time=1.0, pc=2, octave=4, duration=3.0,
                          amplitude=0.2, instrument='church_bell',
                          category='melody')
            return self.events

    piece = MyPiece(seed=42, duration=30, title="My Bell Piece")
    piece.generate()

    renderer = Renderer()
    result = renderer.render_and_save(piece, "my_bell_piece")
"""

from .composition import Composition
from .renderer import Renderer
from .seed_manager import SeedManager
from .event import SynthEvent
from .registry import InstrumentRegistry
from .config import (
    SAMPLE_RATE, BASE_FREQ, NOTE_NAMES,
    freq_from_pitch_class, pitch_to_name, pitch_to_midi,
)

__all__ = [
    'Composition',
    'Renderer',
    'SeedManager',
    'SynthEvent',
    'InstrumentRegistry',
    'SAMPLE_RATE',
    'BASE_FREQ',
    'NOTE_NAMES',
    'freq_from_pitch_class',
    'pitch_to_name',
    'pitch_to_midi',
]

__version__ = '0.1.0'
