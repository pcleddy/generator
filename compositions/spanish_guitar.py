"""
spanish_guitar.py — Spanish/flamenco-style piece for Karplus-Strong nylon guitar.

Phrygian mode (E Phrygian: E F G A B C D), the defining sound of flamenco.
Rasgueado strumming (rapid downward sweeps across strings), picado runs
(fast single-note scale passages), and tremolo (rapid repetition of one note).

Structure (30s at ~100 BPM):
  0-8s:   Rasgueado intro — Am → G → F → E chords (Andalusian cadence)
  8-18s:  Picado melody — fast Phrygian runs over the progression
  18-24s: Tremolo section — sustained melody note with rapid repetition
  24-30s: Final rasgueado flourish → E major chord (Picardy third)

Tempo: ~100 BPM (moderate flamenco feel, not too fast).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis_engine import Renderer, SeedManager, SynthEvent, InstrumentRegistry
from synthesis_engine.config import SAMPLE_RATE

TEMPO = 100
BEAT = 60.0 / TEMPO
EIGHTH = BEAT / 2
SIXTEENTH = BEAT / 4

# E Phrygian: E F G A B C D
# pc: E=4, F=5, G=7, A=9, B=11, C=0, D=2
PHRYGIAN = [4, 5, 7, 9, 11, 0, 2]

# Chord voicings (guitar-style, low to high)
# Each chord: list of (pitch_class, octave) tuples
CHORDS = {
    "Am":  [(9,2), (4,3), (9,3), (0,4), (4,4)],
    "G":   [(7,2), (11,2), (2,3), (7,3), (11,3), (2,4)],
    "F":   [(5,2), (0,3), (5,3), (9,3), (0,4)],
    "E":   [(4,2), (11,2), (4,3), (8,3), (11,3), (4,4)],   # E major (with G#)
    "Em":  [(4,2), (11,2), (4,3), (7,3), (11,3), (4,4)],
    "Dm":  [(2,2), (9,2), (2,3), (5,3), (9,3)],
    "C":   [(0,3), (4,3), (7,3), (0,4), (4,4)],
    "Bb":  [(10,2), (5,3), (10,3), (2,4)],
}

# Andalusian cadence: Am → G → F → E (the heart of flamenco)
CADENCE = ["Am", "G", "F", "E"]


def rasgueado(events, chord_name, start_time, strum_speed=0.03, amp=0.5, rng=None):
    """Rasgueado: rapid downward strum across all strings.

    Each string sounds slightly after the previous one (strum_speed gap).
    """
    chord = CHORDS[chord_name]
    for i, (pc, octave) in enumerate(chord):
        t = start_time + i * strum_speed
        # Each string rings for a while
        dur = BEAT * 2.5 - i * strum_speed
        # Slight amp variation — louder in the middle
        a = amp * (0.8 + 0.4 * (1 - abs(i - len(chord)/2) / len(chord)))
        # Slight timing humanization
        if rng:
            t += rng.uniform(-0.005, 0.005)
            a *= rng.uniform(0.9, 1.1)
        events.append(SynthEvent(
            time=t, pitch_class=pc, octave=octave,
            duration=max(dur, 0.3), amplitude=min(a, 0.7),
            instrument="guitar_nylon",
            category="rasgueado", section="strum",
        ))


def golpe_strum(events, chord_name, start_time, strum_speed=0.015, amp=0.55, rng=None):
    """Golpe: sharp percussive strum (faster, louder, shorter duration)."""
    chord = CHORDS[chord_name]
    for i, (pc, octave) in enumerate(chord):
        t = start_time + i * strum_speed
        dur = BEAT * 0.8
        a = amp * rng.uniform(0.9, 1.1) if rng else amp
        events.append(SynthEvent(
            time=t, pitch_class=pc, octave=octave,
            duration=dur, amplitude=min(a, 0.7),
            instrument="guitar_nylon",
            category="golpe", section="strum",
        ))


def picado_run(events, start_time, notes, note_dur=None, base_amp=0.45, rng=None):
    """Picado: fast alternating-finger scale run.

    notes: list of (pitch_class, octave) tuples
    """
    if note_dur is None:
        note_dur = SIXTEENTH
    t = start_time
    for i, (pc, octave) in enumerate(notes):
        dur = note_dur * rng.uniform(0.8, 1.0) if rng else note_dur
        # Alternating finger emphasis (index/middle)
        amp = base_amp * (1.1 if i % 2 == 0 else 0.9)
        if rng:
            amp *= rng.uniform(0.9, 1.1)
        events.append(SynthEvent(
            time=t, pitch_class=pc, octave=octave,
            duration=dur * 0.85,  # slight gap between notes
            amplitude=min(amp, 0.65),
            instrument="guitar_nylon",
            category="picado", section="melody",
        ))
        t += dur
    return t


def tremolo(events, start_time, pc, octave, total_dur, repetition_speed=0.08, amp=0.4, rng=None):
    """Tremolo: rapid repetition of a single note (p-a-m-i finger pattern)."""
    t = start_time
    end = start_time + total_dur
    i = 0
    while t < end:
        # p-a-m-i pattern: thumb louder, fingers lighter
        finger = i % 4
        if finger == 0:
            a = amp * 1.15  # thumb
        else:
            a = amp * rng.uniform(0.85, 1.0) if rng else amp * 0.9
        dur = repetition_speed * rng.uniform(0.9, 1.1) if rng else repetition_speed
        events.append(SynthEvent(
            time=t, pitch_class=pc, octave=octave,
            duration=dur * 0.9,
            amplitude=min(a, 0.6),
            instrument="guitar_nylon",
            category="tremolo", section="tremolo",
        ))
        t += dur
        i += 1


def generate():
    rng = SeedManager(42)
    events = []

    # ════════════════════════════════════════════════════════════
    # Section 1: RASGUEADO INTRO (0-8s)
    # Andalusian cadence: Am → G → F → E, twice
    # ════════════════════════════════════════════════════════════
    t = 0.1
    for rep in range(2):
        for chord_name in CADENCE:
            rasgueado(events, chord_name, t, strum_speed=0.025, amp=0.5, rng=rng)
            t += BEAT * 2  # 2 beats per chord

    # ════════════════════════════════════════════════════════════
    # Section 2: PICADO RUNS (8-18s)
    # Fast Phrygian runs with the cadence underneath
    # ════════════════════════════════════════════════════════════

    # Background: softer rasgueado chords
    bg_t = 8.0
    for chord_name in CADENCE + CADENCE + ["Am", "G"]:
        rasgueado(events, chord_name, bg_t, strum_speed=0.03, amp=0.25, rng=rng)
        bg_t += BEAT * 1.5

    # Picado melody runs
    t = 8.2

    # Run 1: ascending E Phrygian from E3 to E4
    run1 = [(4,3), (5,3), (7,3), (9,3), (11,3), (0,4), (2,4), (4,4)]
    t = picado_run(events, t, run1, SIXTEENTH, 0.5, rng)
    t += EIGHTH  # brief pause

    # Run 2: descending from high, faster
    run2 = [(4,4), (2,4), (0,4), (11,3), (9,3), (7,3), (5,3), (4,3)]
    t = picado_run(events, t, run2, SIXTEENTH * 0.8, 0.5, rng)
    t += BEAT * 0.5

    # Run 3: chromatic approach figure (very flamenco)
    # E-F-E-D-E with ornament
    run3 = [(4,4), (5,4), (4,4), (2,4), (4,4)]
    t = picado_run(events, t, run3, SIXTEENTH * 0.7, 0.55, rng)
    t += EIGHTH

    # Run 4: longer ascending run with chromatic passing tones
    run4 = [(4,3), (5,3), (6,3), (7,3), (9,3), (10,3), (11,3),
            (0,4), (1,4), (2,4), (4,4), (5,4), (7,4)]
    t = picado_run(events, t, run4, SIXTEENTH * 0.75, 0.5, rng)
    t += BEAT * 0.3

    # Run 5: fast descending cascade
    run5 = [(7,4), (5,4), (4,4), (2,4), (0,4), (11,3), (9,3),
            (7,3), (5,3), (4,3)]
    t = picado_run(events, t, run5, SIXTEENTH * 0.65, 0.5, rng)
    t += BEAT * 0.5

    # Run 6: repeated-note ornament (alzapua-like)
    for _ in range(3):
        run6 = [(4,4), (4,4), (5,4), (4,4)]
        t = picado_run(events, t, run6, SIXTEENTH * 0.6, 0.5, rng)
        t += SIXTEENTH

    # ════════════════════════════════════════════════════════════
    # Section 3: TREMOLO (18-24s)
    # Sustained melody over gentle chords
    # ════════════════════════════════════════════════════════════

    # Background chords — very soft
    bg_t = 18.0
    for chord_name in ["Am", "Dm", "Bb", "E"]:
        rasgueado(events, chord_name, bg_t, strum_speed=0.04, amp=0.18, rng=rng)
        bg_t += BEAT * 2.5

    # Tremolo melody: B4 (longing) → C5 → A4 → E4
    tremolo(events, 18.2, 11, 4, BEAT * 3, 0.07, 0.4, rng)   # B4
    tremolo(events, 20.0, 0, 5,  BEAT * 2, 0.07, 0.38, rng)   # C5
    tremolo(events, 21.5, 9, 4,  BEAT * 2.5, 0.07, 0.42, rng) # A4
    tremolo(events, 23.0, 4, 4,  BEAT * 2, 0.075, 0.35, rng)  # E4 (resolve)

    # ════════════════════════════════════════════════════════════
    # Section 4: FINAL FLOURISH (24-30s)
    # Big rasgueado cadence → final E major chord
    # ════════════════════════════════════════════════════════════

    # Fast descending picado into the final cadence
    run_final = [(4,4), (2,4), (0,4), (11,3), (9,3), (7,3), (5,3), (4,3),
                 (2,3), (0,3), (11,2), (9,2)]
    t = picado_run(events, 24.2, run_final, SIXTEENTH * 0.55, 0.55, rng)

    # Cadence: Am → G → F → E with golpe strums
    t = 25.5
    for chord_name in ["Am", "G", "F"]:
        golpe_strum(events, chord_name, t, strum_speed=0.02, amp=0.55, rng=rng)
        t += BEAT * 1.2

    # Final E major — big rasgueado, slower strum for drama
    rasgueado(events, "E", t, strum_speed=0.04, amp=0.65, rng=rng)

    # Let it ring...
    total_dur = 32.0

    return events, total_dur


def main():
    events, duration = generate()

    print(f"Spanish Guitar: {len(events)} events, {duration:.1f}s")

    # Count by section
    sections = {}
    for e in events:
        sections[e.category] = sections.get(e.category, 0) + 1
    for sec, n in sorted(sections.items()):
        print(f"  {sec}: {n} events")

    registry = InstrumentRegistry()
    renderer = Renderer(registry=registry)
    rng = SeedManager(42)

    audio = renderer.render(events, duration, rng=rng, reverb_preset="intimate")

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "spanish_guitar.wav")
    mp3_path = os.path.join(out_dir, "audio", "spanish_guitar.mp3")
    json_path = os.path.join(out_dir, "scores", "spanish_guitar.json")

    renderer.save_wav(audio, wav_path)
    renderer.save_mp3(wav_path, mp3_path)
    print(f"  Written: {mp3_path}")

    import json
    with open(json_path, "w") as f:
        json.dump({
            "title": "Spanish Guitar — E Phrygian",
            "duration": duration,
            "events": [e.to_dict() for e in events],
            "audio_file": "spanish_guitar.mp3",
            "seed": 42,
        }, f, indent=2)
    print(f"  Written: {json_path}")


if __name__ == "__main__":
    main()
