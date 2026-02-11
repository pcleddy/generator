"""
config.py — Global constants for the synthesis engine.

One definition of each. Import from here, not from webern_pointillism.
"""

SAMPLE_RATE = 44100
BASE_FREQ = 261.63  # C4 (middle C)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def freq_from_pitch_class(pc, octave):
    """Convert pitch class (0-11) and octave to frequency in Hz."""
    return BASE_FREQ * (2 ** ((pc - 0) / 12 + (octave - 4)))


def freq_from_pc_micro(pc, octave, rng, detune_range=30):
    """Frequency with microtonal detuning. detune_range in cents."""
    base = freq_from_pitch_class(pc, octave)
    detune_cents = rng.uniform(-detune_range, detune_range)
    return base * (2 ** (detune_cents / 1200.0)), detune_cents


def pitch_to_name(pc, octave):
    """Convert pitch class + octave to note name (e.g. 'C4', 'F#5')."""
    return f"{NOTE_NAMES[pc % 12]}{octave}"


def midi_to_freq(midi_note):
    """MIDI note number to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def pitch_to_midi(pc, octave):
    """Pitch class + octave to MIDI note number."""
    return pc + (octave + 1) * 12
