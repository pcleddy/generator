"""
timbre_gallery.py — Generate a demo file showcasing all instruments in the synthesis_engine.

Each instrument plays 3 ascending notes so you can hear its character.
Grouped by category with silence between groups.

Skips voice instruments (they need formant filtering, not in renderer yet).

Usage:
    cd generator
    python tools/timbre_gallery.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis_engine import Composition, Renderer, SeedManager, SynthEvent, InstrumentRegistry
from synthesis_engine.config import SAMPLE_RATE
from synthesis_engine.profiles.strings import STRING_PROFILES

# Fix the bass collision: voice "bass" overwrites string "bass" in the registry.
# Re-register the string bass under its own name.
registry = InstrumentRegistry()
registry.register("string_bass", {**STRING_PROFILES["bass"], "name": "string_bass"})


# ── Instrument gallery definitions ──────────────────────────────────────
# Each entry: (instrument_name, display_label, [(pitch_class, octave), ...], note_duration)

GALLERY = [
    # ── BELLS ──
    ("glockenspiel",    "Glockenspiel",         [(0,6), (4,6), (7,6)],     1.8),
    ("celesta",         "Celesta",              [(0,5), (4,5), (7,5)],     2.0),
    ("tubular_bell",    "Tubular Bell",         [(2,3), (6,3), (9,3)],     2.5),
    ("church_bell",     "Church Bell",          [(2,3), (9,3), (2,4)],     3.0),
    ("wind_chime",      "Wind Chime",           [(7,6), (11,6), (2,7)],    1.5),
    ("papa_bell",       "Papa Bell (10-partial)",[(2,2), (7,2), (2,3)],     3.0),
    ("music_box_tine",  "Music Box Tine",       [(7,5), (11,5), (2,6)],    1.5),
    ("music_box_worn",  "Music Box (Worn)",     [(7,5), (11,5), (2,6)],    1.5),

    # ── WOOD ──
    ("wood_xylophone",  "Wood Xylophone",       [(0,5), (4,5), (7,5)],     1.5),
    ("marimba",         "Marimba",              [(0,4), (4,4), (7,4)],     2.0),

    # ── STRINGS (webern timbres) ──
    ("cello_pont",      "Cello sul Ponticello", [(0,3), (4,3), (7,3)],     2.0),
    ("cello_tasto",     "Cello sul Tasto",      [(0,3), (4,3), (7,3)],     2.0),
    ("pizzicato",       "Pizzicato",            [(0,3), (4,3), (7,3)],     1.2),

    # ── STRINGS (orchestral) ──
    ("violin",          "Violin",               [(7,4), (11,4), (2,5)],    2.0),
    ("viola",           "Viola",                [(0,4), (4,4), (7,4)],     2.0),
    ("cello",           "Cello",                [(0,3), (4,3), (7,3)],     2.0),
    ("string_bass",     "String Bass",          [(0,2), (4,2), (7,2)],     2.0),

    # ── WINDS ──
    ("flute_breathy",       "Flute (breathy)",      [(7,4), (11,4), (2,5)],    2.0),
    ("clarinet_chalumeau",  "Clarinet (chalumeau)", [(2,3), (5,3), (9,3)],     2.0),
    ("oboe_pp",             "Oboe pp",              [(0,4), (4,4), (7,4)],     2.0),

    # ── PITCHED PERCUSSION ──
    ("bell_struck",     "Bell (struck)",        [(0,5), (4,5), (7,5)],     2.0),
    ("glass_harmonica", "Glass Harmonica",      [(7,4), (11,4), (2,5)],    2.5),

    # ── BRASS ──
    ("horn",            "Horn",                 [(5,3), (9,3), (0,4)],     2.0),
    ("trumpet",         "Trumpet",              [(0,4), (4,4), (7,4)],     1.8),
    ("trombone",        "Trombone",             [(10,2), (2,3), (5,3)],    2.0),
]


def generate_gallery():
    """Generate the timbre gallery as events + render."""
    rng = SeedManager(42)
    events = []

    current_time = 0.5  # start with a brief lead-in

    for instrument, label, notes, note_dur in GALLERY:
        # Small gap before each instrument
        note_gap = 0.3

        for i, (pc, octave) in enumerate(notes):
            # Slight amplitude variation — middle note louder
            amp = 0.7 if i == 1 else 0.55

            events.append(SynthEvent(
                time=current_time,
                pitch_class=pc,
                octave=octave,
                duration=note_dur,
                amplitude=amp,
                instrument=instrument,
                category=label,
                section=label,
            ))
            current_time += note_dur + note_gap

        # Gap between instruments
        current_time += 1.5

    total_duration = current_time + 2.0  # 2s tail for reverb

    print(f"Timbre Gallery: {len(events)} events, {total_duration:.1f}s")
    print(f"  {len(GALLERY)} instruments across {len(set(g[0] for g in GALLERY))} timbres")

    # Render
    renderer = Renderer(registry=registry)
    audio = renderer.render(events, total_duration, rng=rng, reverb_preset="room")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    wav_path = os.path.join(out_dir, "timbre_gallery.wav")
    mp3_path = os.path.join(out_dir, "timbre_gallery.mp3")
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "scores", "timbre_gallery.json")

    renderer.save_wav(audio, wav_path)
    print(f"  Written: {wav_path}")

    renderer.save_mp3(wav_path, mp3_path)
    print(f"  Written: {mp3_path}")

    # Save events JSON for the animated player
    import json
    event_dicts = [e.to_dict() for e in events]
    json_data = {
        "title": "Timbre Gallery — All Instruments",
        "duration": total_duration,
        "events": event_dicts,
        "audio_file": "timbre_gallery.mp3",
        "seed": 42,
        "instrument_order": [label for _, label, _, _ in GALLERY],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Written: {json_path}")

    # Print timing index
    print("\n── Timing Index ──")
    t = 0.5
    for instrument, label, notes, note_dur in GALLERY:
        mins = int(t // 60)
        secs = t % 60
        print(f"  {mins}:{secs:05.2f}  {label} ({instrument})")
        for _ in notes:
            t += note_dur + 0.3
        t += 1.5


if __name__ == "__main__":
    generate_gallery()
