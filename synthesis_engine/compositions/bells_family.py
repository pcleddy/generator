"""
bells_family.py — Bells Family: Rude Guy, Boy, Papa & Little Sister

Four bell players collide in overlapping chaos.
Ported from cage_bells_family.py to the OOP engine.

Usage:
    from synthesis_engine.compositions.bells_family import BellsFamily
    from synthesis_engine import Renderer

    piece = BellsFamily(seed=42)
    piece.generate()
    Renderer().render_and_save(piece, "cage_bells_family")
"""

import numpy as np
from scipy.io import wavfile

from ..composition import Composition
from ..config import SAMPLE_RATE, freq_from_pitch_class
from ..synthesis.bell import bell_strike
from ..profiles.bells import BELL_PROFILES


class BellsFamily(Composition):
    """Four bell players in 30 seconds of overlapping chaos."""

    def __init__(self, seed=42, duration=30.0, **params):
        super().__init__(
            seed=seed, duration=duration,
            title="Bells Family — Rude Guy, Boy, Papa & Little Sister",
            reverb_preset="room",
            **params,
        )

    def generate(self):
        """Generate all four layers."""
        self.events = []
        self._generate_rude_guy()
        self._generate_boy()
        self._generate_papa()
        self._generate_sister()
        return self.events

    def _generate_rude_guy(self):
        """One rude guy — varies pitch, plays fragments, rhythm variety."""
        current_time = self.rng.uniform(0.3, 0.8)

        while current_time < self.duration - 1.0:
            mode = self.rng.choice(
                ['quick', 'cruising', 'lazy'],
            )
            # Weight it manually since we don't have p= in choice
            r = self.rng.random()
            if r < 0.25:
                mode = 'quick'
            elif r < 0.75:
                mode = 'cruising'
            else:
                mode = 'lazy'

            pitch_shift = self.rng.uniform(0.93, 1.03)
            frag_dur = self.rng.uniform(1.0, 3.5)

            pc = int(pitch_shift * 12) % 12
            self.add_event(
                time=current_time,
                pc=pc, octave=4,
                duration=frag_dur,
                amplitude=0.35,
                instrument='cluster',
                category='rude_guy',
            )

            if mode == 'quick':
                gap = self.rng.uniform(1.5, 2.5)
            elif mode == 'cruising':
                gap = self.rng.uniform(3.0, 4.5)
            else:
                gap = self.rng.uniform(5.0, 7.0)

            current_time += frag_dur + gap

    def _generate_boy(self):
        """Boy with F#5 church bell — 18 clangs, double-hits."""
        current_time = self.rng.uniform(0.5, 1.2)
        clang_count = 0
        max_clangs = 18

        while clang_count < max_clangs and current_time < self.duration - 0.5:
            amp = self.rng.uniform(0.12, 0.22)
            dur = self.rng.uniform(2.5, 4.0)

            self.add_event(
                time=current_time,
                pc=6, octave=5,  # F#5
                duration=dur,
                amplitude=amp,
                instrument='church_bell',
                category='boy',
            )
            clang_count += 1

            # Double hit 20%
            if self.rng.random() < 0.2 and clang_count < max_clangs:
                double_gap = self.rng.uniform(0.08, 0.15)
                current_time += double_gap
                amp2 = amp * self.rng.uniform(0.6, 0.9)

                self.add_event(
                    time=current_time,
                    pc=6, octave=5,
                    duration=dur * 0.8,
                    amplitude=amp2,
                    instrument='church_bell',
                    category='boy',
                )
                clang_count += 1

            gap = self.rng.uniform(1.0, 2.5)
            current_time += gap

    def _generate_papa(self):
        """Papa on D2 — syncopated, loud, double-whack 35%."""
        current_time = self.rng.uniform(1.0, 2.0)
        toll_count = 0
        max_tolls = 22

        while toll_count < max_tolls and current_time < self.duration - 0.5:
            amp = self.rng.uniform(0.25, 0.40)
            dur = self.rng.uniform(3.5, 5.0)

            self.add_event(
                time=current_time,
                pc=2, octave=2,  # D2
                duration=dur,
                amplitude=amp,
                instrument='papa_bell',
                category='papa',
            )
            toll_count += 1

            # Double whack 35%
            if self.rng.random() < 0.35 and toll_count < max_tolls:
                double_gap = self.rng.uniform(0.12, 0.25)
                current_time += double_gap
                amp2 = amp * self.rng.uniform(0.7, 1.0)

                self.add_event(
                    time=current_time,
                    pc=2, octave=2,
                    duration=dur * 0.9,
                    amplitude=amp2,
                    instrument='papa_bell',
                    category='papa',
                )
                toll_count += 1

            gap = self.rng.uniform(0.8, 2.0)
            current_time += gap

    def _generate_sister(self):
        """Little sister on B5 glockenspiel — 42 rapid-fire dings."""
        current_time = self.rng.uniform(0.3, 0.7)
        ding_count = 0
        max_dings = 42

        while ding_count < max_dings and current_time < self.duration - 0.3:
            amp = self.rng.uniform(0.08, 0.18)
            dur = self.rng.uniform(1.5, 2.5)

            self.add_event(
                time=current_time,
                pc=11, octave=5,  # B5
                duration=dur,
                amplitude=amp,
                instrument='glockenspiel',
                category='sister',
            )
            ding_count += 1

            # Rapid-fire burst 25%
            if self.rng.random() < 0.25 and ding_count < max_dings - 2:
                burst = self.rng.randint(2, 3)
                for _ in range(burst):
                    burst_gap = self.rng.uniform(0.15, 0.4)
                    current_time += burst_gap
                    if current_time >= self.duration - 0.3:
                        break
                    burst_amp = amp * self.rng.uniform(0.7, 1.1)

                    self.add_event(
                        time=current_time,
                        pc=11, octave=5,
                        duration=dur * 0.7,
                        amplitude=burst_amp,
                        instrument='glockenspiel',
                        category='sister',
                    )
                    ding_count += 1

            # Double-tap 20%
            elif self.rng.random() < 0.20 and ding_count < max_dings:
                tap_gap = self.rng.uniform(0.06, 0.12)
                current_time += tap_gap
                tap_amp = amp * self.rng.uniform(0.5, 0.8)

                self.add_event(
                    time=current_time,
                    pc=11, octave=5,
                    duration=dur * 0.6,
                    amplitude=tap_amp,
                    instrument='glockenspiel',
                    category='sister',
                )
                ding_count += 1

            gap = self.rng.uniform(0.3, 1.2)
            current_time += gap
