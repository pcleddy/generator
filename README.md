# Impossible Sound Generator

Algorithmic synthesis experiments — generating sound from code with no samples, no DAW, no MIDI. Pure math into `.wav`.

Built with Python, NumPy, and SciPy. Nothing else.

## What's here

**`ambient_synthesis.py`** — Beatless ambient soundscapes. Five layers: primary 12-tone row, inverted row, microtonal chaos (40% anchored, 60% drift), sub-bass with LFO, and cluster collisions. 120 seconds of slow evolution.

**`webern_pointillism.py`** — Anton Webern-inspired pointillism. Isolated tones separated by silence, wide register leaps, derived row from trichord `[0, 1, 4]`. 8 instrument timbres with per-partial envelopes, attack transients, pitch micro-drift, delayed vibrato, and inharmonicity modeling. Also serves as the shared synthesis engine.

**`berg_lyrical.py`** — Alban Berg-inspired lyricism. The opposite of the Webern: lush overlapping legato lines, triadic row (Violin Concerto concept), waltz fragments, five simultaneous voices building to a G minor tonal window. 90 seconds.

**`berg_vocal.py`** — Choral-orchestral piece with vocal synthesis. Formant-based voice simulation: glottal pulse source through all-pole resonator cascade, vowel morphing, four voice types (soprano/alto/tenor/bass), irregular vibrato, jitter, shimmer. Instruments get body resonance filtering. 100 seconds, five sections.

**`sound_demo.py`** — Instrument timbre gallery. Each of 8 timbres sustained at A3 (220 Hz) for evaluation, plus a register sweep across A2–A6.

**`voice_demo.py`** — Vocal synthesis test bench. Voice type gallery, vowel sweep (ah/ee/oh/oo/eh/mm), choir chord, and unison comparison.

## Quick start

```bash
pip install numpy scipy

# Generate everything
python ambient_synthesis.py
python webern_pointillism.py --seed 7 --output webern_01.wav
python berg_lyrical.py --seed 13 --output berg_01.wav
python berg_vocal.py --seed 21 --output berg_vocal_01.wav
python sound_demo.py
python voice_demo.py
```

Seeds are optional — omit for randomized output. Different seeds = different pieces from the same compositional logic.

## Synthesis architecture

Two signal chains:

**Instruments** (`webern_pointillism.py`): Additive synthesis with per-partial envelopes (upper harmonics decay faster), string stiffness inharmonicity, filtered noise transients (bow/breath/click), pitch micro-drift via bounded random walk, delayed vibrato with rate wobble. Optional body resonance filtering.

**Voices** (`berg_vocal.py`): Glottal pulse source (Rosenberg model with spectral tilt) through cascaded all-pole resonators modeling the vocal tract. Formant frequencies from Peterson & Barney/Fant measurements, shifted per voice type. Overlap-add vowel morphing, aspiration noise, jitter/shimmer.

## Project policy

See `AMBIENT_PROJECT_POLICY.md` for parameter tuning guidelines, modification workflow, and documentation standards.

## Status

Early POC. The instrument timbres are serviceable; the vocal synthesis is experimental and actively evolving.
