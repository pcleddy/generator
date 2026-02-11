"""
new_synth_demo.py — Demo showcasing the new synthesis methods:
  1. Karplus-Strong plucked strings (8 instruments)
  2. FM brass (6 instruments)
  3. Side-by-side comparison: old additive brass vs new FM brass

Usage:
    cd generator
    python tools/new_synth_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis_engine import Renderer, SeedManager, SynthEvent, InstrumentRegistry
from synthesis_engine.config import SAMPLE_RATE

registry = InstrumentRegistry()

# ── Demo definitions ──

PLUCKED_DEMO = [
    # (instrument, label, notes [(pc, octave)...], duration)
    ("guitar_nylon",  "Nylon Guitar",     [(4,3), (7,3), (0,4), (4,4)],  2.5),
    ("guitar_steel",  "Steel Guitar",     [(4,3), (7,3), (0,4), (4,4)],  2.5),
    ("harp",          "Harp",             [(0,4), (4,4), (7,4), (0,5)],  2.0),
    ("plucked_cello", "Plucked Cello",    [(0,3), (4,3), (7,3), (0,4)],  3.0),
    ("banjo",         "Banjo",            [(7,3), (0,4), (2,4), (7,4)],  1.5),
    ("sitar",         "Sitar",            [(2,3), (7,3), (9,3), (2,4)],  2.5),
    ("koto",          "Koto",             [(0,4), (2,4), (5,4), (9,4)],  2.0),
    ("harpsichord",   "Harpsichord",      [(0,4), (4,4), (7,4), (0,5)],  1.8),
]

FM_BRASS_DEMO = [
    ("fm_trumpet",       "FM Trumpet",        [(0,4), (4,4), (7,4)],  2.5),
    ("fm_horn",          "FM French Horn",     [(5,3), (9,3), (0,4)],  3.0),
    ("fm_trombone",      "FM Trombone",        [(10,2), (2,3), (5,3)], 3.0),
    ("fm_tuba",          "FM Tuba",            [(5,2), (0,2), (7,2)],  3.5),
    ("fm_flugelhorn",    "FM Flugelhorn",      [(0,4), (4,4), (7,4)],  2.5),
    ("fm_muted_trumpet", "FM Muted Trumpet",   [(0,4), (4,4), (7,4)],  2.0),
]

# Side-by-side: old additive brass vs new FM brass
COMPARISON = [
    # Old additive versions first, then FM version
    ("trumpet",    "Old Trumpet (additive)",  [(0,4), (4,4), (7,4)], 2.0),
    ("fm_trumpet", "NEW Trumpet (FM)",        [(0,4), (4,4), (7,4)], 2.5),
    ("horn",       "Old Horn (additive)",      [(5,3), (9,3), (0,4)], 2.0),
    ("fm_horn",    "NEW Horn (FM)",            [(5,3), (9,3), (0,4)], 3.0),
    ("trombone",   "Old Trombone (additive)",  [(10,2), (2,3), (5,3)], 2.0),
    ("fm_trombone","NEW Trombone (FM)",        [(10,2), (2,3), (5,3)], 3.0),
]


def build_events(sections, start_time=0.5):
    """Build SynthEvent list from section definitions."""
    events = []
    t = start_time

    for instrument, label, notes, note_dur in sections:
        note_gap = 0.25
        for i, (pc, octave) in enumerate(notes):
            amp = 0.65 if i == len(notes) // 2 else 0.5
            events.append(SynthEvent(
                time=t,
                pitch_class=pc,
                octave=octave,
                duration=note_dur,
                amplitude=amp,
                instrument=instrument,
                category=label,
                section=label,
            ))
            t += note_dur + note_gap
        t += 1.5  # gap between instruments

    return events, t


def generate_demo():
    rng = SeedManager(42)
    renderer = Renderer(registry=registry)
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Part 1: Plucked strings ──
    print("=== Part 1: Karplus-Strong Plucked Strings ===")
    plucked_events, plucked_end = build_events(PLUCKED_DEMO)
    plucked_dur = plucked_end + 2.0
    print(f"  {len(plucked_events)} events, {plucked_dur:.1f}s")

    rng_plucked = SeedManager(42)
    audio_plucked = renderer.render(plucked_events, plucked_dur, rng=rng_plucked, reverb_preset="room")
    wav_p = os.path.join(out_dir, "audio", "demo_plucked_strings.wav")
    mp3_p = os.path.join(out_dir, "audio", "demo_plucked_strings.mp3")
    renderer.save_wav(audio_plucked, wav_p)
    renderer.save_mp3(wav_p, mp3_p)
    print(f"  Written: {mp3_p}")

    # Save JSON for player
    import json
    json_p = os.path.join(out_dir, "scores", "demo_plucked_strings.json")
    with open(json_p, "w") as f:
        json.dump({
            "title": "Karplus-Strong Plucked Strings",
            "duration": plucked_dur,
            "events": [e.to_dict() for e in plucked_events],
            "audio_file": "demo_plucked_strings.mp3",
            "seed": 42,
        }, f, indent=2)

    # ── Part 2: FM Brass ──
    print("\n=== Part 2: FM Brass ===")
    fm_events, fm_end = build_events(FM_BRASS_DEMO)
    fm_dur = fm_end + 2.0
    print(f"  {len(fm_events)} events, {fm_dur:.1f}s")

    rng_fm = SeedManager(42)
    audio_fm = renderer.render(fm_events, fm_dur, rng=rng_fm, reverb_preset="concert_hall")
    wav_f = os.path.join(out_dir, "audio", "demo_fm_brass.wav")
    mp3_f = os.path.join(out_dir, "audio", "demo_fm_brass.mp3")
    renderer.save_wav(audio_fm, wav_f)
    renderer.save_mp3(wav_f, mp3_f)
    print(f"  Written: {mp3_f}")

    json_f = os.path.join(out_dir, "scores", "demo_fm_brass.json")
    with open(json_f, "w") as f:
        json.dump({
            "title": "FM Brass Synthesis",
            "duration": fm_dur,
            "events": [e.to_dict() for e in fm_events],
            "audio_file": "demo_fm_brass.mp3",
            "seed": 42,
        }, f, indent=2)

    # ── Part 3: Comparison ──
    print("\n=== Part 3: Old vs New Brass Comparison ===")
    comp_events, comp_end = build_events(COMPARISON)
    comp_dur = comp_end + 2.0
    print(f"  {len(comp_events)} events, {comp_dur:.1f}s")

    rng_comp = SeedManager(42)
    audio_comp = renderer.render(comp_events, comp_dur, rng=rng_comp, reverb_preset="chamber")
    wav_c = os.path.join(out_dir, "audio", "demo_brass_comparison.wav")
    mp3_c = os.path.join(out_dir, "audio", "demo_brass_comparison.mp3")
    renderer.save_wav(audio_comp, wav_c)
    renderer.save_mp3(wav_c, mp3_c)
    print(f"  Written: {mp3_c}")

    json_c = os.path.join(out_dir, "scores", "demo_brass_comparison.json")
    with open(json_c, "w") as f:
        json.dump({
            "title": "Brass: Additive vs FM Comparison",
            "duration": comp_dur,
            "events": [e.to_dict() for e in comp_events],
            "audio_file": "demo_brass_comparison.mp3",
            "seed": 42,
        }, f, indent=2)

    # ── Timing indexes ──
    print("\n── Plucked Strings Timing ──")
    _print_index(PLUCKED_DEMO)
    print("\n── FM Brass Timing ──")
    _print_index(FM_BRASS_DEMO)
    print("\n── Comparison Timing ──")
    _print_index(COMPARISON)


def _print_index(sections, start=0.5):
    t = start
    for instrument, label, notes, note_dur in sections:
        m, s = int(t // 60), t % 60
        print(f"  {m}:{s:05.2f}  {label}")
        for _ in notes:
            t += note_dur + 0.25
        t += 1.5


if __name__ == "__main__":
    generate_demo()
