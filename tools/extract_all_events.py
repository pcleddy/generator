"""
extract_all_events.py — Extract event data from ALL compositions

This script:
1. Imports and runs generation functions from each composition file
2. Extracts/normalizes event data (time, pitch class, octave, duration, amplitude, type)
3. Saves as JSON files (events_{piece_name}.json)
4. Generates static PNG scores using dark background, time vs pitch visualization
   with shapes/colors per timbre type

Compositions processed:
  - webern_pointillism.py: generate_webern_piece(seed=None)
  - berg_lyrical.py: generate_berg_piece(seed=None)
  - berg_extended.py: generate_berg_extended(seed=42)
  - cage_ambient.py: extract_cage_events(seed=99) via score_generator.py
  - bells_bergman.py: generate_bells_piece(seed=42)
  - bells_pizz.py: generate_bells_pizz(seed=77)
  - bells_gentle.py: generate_gentle_bells(seed=55)
  - tubular_low.py: generate_tubular_low(seed=31)
  - one_more_thing.py: skipped (complex, audio only)

Output:
  - events_{piece_name}.json with normalized event structure
  - {piece_name}_score.png with graphical visualization
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import random

# Add current directory to path for imports
sys.path.insert(0, '/sessions/sweet-nice-volta/mnt/generator')

# =====================================================================
# NOTE NAMING UTILITY
# =====================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_to_midi(pitch_class, octave):
    """Convert pitch class (0-11) + octave to MIDI note number."""
    return pitch_class + (octave + 1) * 12

def midi_to_name(midi_note):
    """Convert MIDI note to readable name."""
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES[pc]}{octave}"


# =====================================================================
# EVENT EXTRACTION FUNCTIONS
# =====================================================================

def extract_webern_events(seed=None):
    """Extract events from Webern pointillism piece."""
    from webern_pointillism import generate_webern_piece

    print("  Extracting Webern piece...")
    audio, events = generate_webern_piece(seed=seed)

    normalized = []
    for ev in events:
        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(ev["duration"]),
            "amplitude": float(ev["amplitude"]),
            "type": ev["timbre"]["name"],
            "category": "pointillist"
        })

    return normalized, 50.0, None


def extract_berg_lyrical_events(seed=None):
    """Extract events from Berg lyrical piece."""
    from berg_lyrical import generate_berg_piece

    print("  Extracting Berg lyrical piece...")
    audio, events = generate_berg_piece(seed=seed)

    normalized = []
    for ev in events:
        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(ev["duration"]),
            "amplitude": float(ev["amplitude"]),
            "type": ev["timbre"]["name"],
            "category": "lyrical"
        })

    return normalized, 90.0, None


def extract_berg_extended_events(seed=42):
    """Extract events from Berg extended piece (7 minutes)."""
    from berg_extended import generate_berg_extended

    print("  Extracting Berg extended piece...")
    audio, events = generate_berg_extended(seed=seed)

    normalized = []
    for ev in events:
        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(ev["duration"]),
            "amplitude": float(ev["amplitude"]),
            "type": ev["timbre"]["name"],
            "category": "extended"
        })

    return normalized, 420.0, None


def extract_cage_ambient_events(seed=99):
    """Extract events from Cage ambient via score_generator."""
    from score_generator import extract_cage_events

    print("  Extracting Cage ambient events...")
    events = extract_cage_events(seed=seed)

    normalized = []
    for ev in events:
        # Handle both single pitch and multiple pitches
        if 'pitches' in ev and ev['pitches']:
            pitch = ev['pitches'][0] if isinstance(ev['pitches'], list) else ev['pitches']
            # Convert MIDI back to pc/octave
            pc = pitch % 12
            octave = (pitch // 12) - 1
        else:
            pc, octave = 0, 3

        normalized.append({
            "time": float(ev["start"]),
            "pc": int(pc),
            "octave": int(octave),
            "duration": float(ev["duration"]),
            "amplitude": float(ev["amplitude"]),
            "type": ev.get("timbre", ev.get("type", "ambient")),
            "category": ev.get("category", "ambient")
        })

    return normalized, 150.0, None


def extract_bells_bergman_events(seed=42):
    """Extract events from bells/Bergman piece."""
    from bells_bergman import generate_bells_piece

    print("  Extracting bells/Bergman piece...")
    audio, events = generate_bells_piece(seed=seed)

    normalized = []
    for ev in events:
        # Handle both 'dur' and 'duration' keys
        dur = ev.get("duration") or ev.get("dur", 1.0)
        # Handle both 'profile' and 'timbre' keys
        profile = ev.get("profile") or ev.get("timbre", "glockenspiel")
        # Handle both 'amp' and 'amplitude' keys
        amp = ev.get("amplitude") or ev.get("amp", 0.05)

        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(dur),
            "amplitude": float(amp),
            "type": str(profile),
            "category": "bell"
        })

    return normalized, 120.0, None


def extract_bells_pizz_events(seed=77):
    """Extract events from bells pizzicato piece."""
    from bells_pizz import generate_bells_pizz

    print("  Extracting bells pizz piece...")
    audio, events = generate_bells_pizz(seed=seed)

    normalized = []
    for ev in events:
        dur = ev.get("duration") or ev.get("dur", 1.0)
        profile = ev.get("profile") or ev.get("timbre", "glockenspiel")
        amp = ev.get("amplitude") or ev.get("amp", 0.05)

        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(dur),
            "amplitude": float(amp),
            "type": str(profile),
            "category": "bell"
        })

    return normalized, 90.0, None


def extract_bells_gentle_events(seed=55):
    """Extract events from gentle bells piece."""
    from bells_gentle import generate_gentle_bells

    print("  Extracting gentle bells piece...")
    audio, events = generate_gentle_bells(seed=seed)

    normalized = []
    for ev in events:
        dur = ev.get("duration") or ev.get("dur", 1.0)
        profile = ev.get("profile") or ev.get("timbre", "glockenspiel")
        amp = ev.get("amplitude") or ev.get("amp", 0.05)

        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(dur),
            "amplitude": float(amp),
            "type": str(profile),
            "category": "bell"
        })

    return normalized, 120.0, None


def extract_tubular_low_events(seed=31):
    """Extract events from tubular low piece."""
    from tubular_low import generate_tubular_low

    print("  Extracting tubular low piece...")
    audio, events = generate_tubular_low(seed=seed)

    normalized = []
    for ev in events:
        dur = ev.get("duration") or ev.get("dur", 1.0)
        profile = ev.get("profile") or ev.get("timbre", "tubular_bell")
        amp = ev.get("amplitude") or ev.get("amp", 0.05)

        normalized.append({
            "time": float(ev["time"]),
            "pc": int(ev["pc"]),
            "octave": int(ev["octave"]),
            "duration": float(dur),
            "amplitude": float(amp),
            "type": str(profile),
            "category": "bell"
        })

    return normalized, 70.0, None


# =====================================================================
# GRAPHICAL SCORE RENDERING
# =====================================================================

def render_score_png(piece_name, events, duration, output_path):
    """Render a graphical score with dark background, time vs pitch."""

    if not events:
        print(f"    (No events to render for {piece_name})")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    # Color palette by type
    TYPE_COLORS = {
        # Webern/Berg instruments
        "cello_pont": "#4488dd",       # cool blue
        "cello_tasto": "#2266aa",      # darker blue
        "flute_breathy": "#66ddff",    # cyan
        "clarinet_chalumeau": "#4499cc", # lighter blue
        "bell_struck": "#ffaa44",      # orange/amber
        "glass_harmonica": "#aaffcc",  # pale green
        "pizzicato": "#ff66aa",        # magenta
        "oboe_pp": "#ddaaff",          # lavender

        # Bell types
        "glockenspiel": "#66ccff",     # bright ice blue
        "celesta": "#cc99ff",          # soft violet
        "tubular_bell": "#ffaa44",     # warm amber
        "church_bell": "#ff6644",      # deep copper
        "wind_chime": "#aaffcc",       # pale green

        # Ambient/other
        "ambient": "#1a5276",          # deep blue
        "string": "#4488dd",           # blue
        "wind": "#66ddff",             # cyan
    }

    CATEGORY_SHAPES = {
        "pointillist": "D",      # diamond
        "lyrical": "o",          # circle
        "extended": "s",         # square
        "bell": "H",             # hexagon
        "ambient": "^",          # triangle up
        "interruption": "v",     # triangle down
    }

    # Collect all MIDI notes for axis scaling
    all_midi = []
    for ev in events:
        midi = pitch_to_midi(ev["pc"], ev["octave"])
        all_midi.append(midi)

    if all_midi:
        y_min = min(all_midi) - 3
        y_max = max(all_midi) + 3
    else:
        y_min, y_max = 24, 96

    # Plot events
    for ev in events:
        midi = pitch_to_midi(ev["pc"], ev["octave"])
        x = ev["time"]
        width = ev["duration"]

        color = TYPE_COLORS.get(ev["type"], "#888888")
        category = ev.get("category", "lyrical")
        marker = CATEGORY_SHAPES.get(category, "o")

        # Size based on amplitude
        size = max(50, ev["amplitude"] * 3000)
        alpha = min(0.3 + ev["amplitude"] * 5, 0.8)

        # Draw as scatter point
        ax.scatter(
            x, midi,
            s=size, c=color, marker=marker,
            alpha=alpha, zorder=5,
            edgecolors="white", linewidths=0.5
        )

        # Draw duration line if significant
        if width > 0.5:
            ax.plot(
                [x, x + width], [midi, midi],
                color=color, alpha=alpha * 0.6,
                linewidth=2, zorder=4
            )

    # Axes and labels
    ax.set_xlim(-1, duration + 1)
    ax.set_ylim(y_min, y_max)

    # Y-axis: pitch names
    y_ticks = list(range(24, 109, 12))
    y_ticks = [y for y in y_ticks if y_min <= y <= y_max]
    y_labels = [midi_to_name(n) for n in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, color="#aaaaaa")

    for yt in y_ticks:
        ax.axhline(y=yt, color="#222233", linewidth=0.4, alpha=0.3, zorder=0)

    # X-axis: time
    step = max(5, int(duration / 20))
    x_ticks = list(range(0, int(duration) + 1, step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}s" for x in x_ticks], fontsize=9, color="#aaaaaa")

    for xt in x_ticks:
        ax.axvline(x=xt, color="#222233", linewidth=0.3, alpha=0.3, zorder=0)

    ax.set_xlabel("Time (seconds)", fontsize=12, color="#cccccc", labelpad=10)
    ax.set_ylabel("Pitch", fontsize=12, color="#cccccc", labelpad=10)

    title = piece_name.replace("_", " ").title()
    ax.set_title(
        f"{title.upper()} — Graphical Score",
        fontsize=18, fontweight="bold", color="#ffffff", pad=20
    )

    # Footer
    n_events = len(events)
    ax.text(
        duration / 2, y_max + 2,
        f"CloudAutomat Labs  •  {n_events} events  •  {duration:.1f}s  •  "
        "Time vs Pitch • Event size = amplitude",
        ha="center", fontsize=9, color="#666688", style="italic"
    )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"    Saved score: {output_path}")


# =====================================================================
# MAIN EXTRACTION
# =====================================================================

def main():
    print("=" * 70)
    print("EXTRACTING EVENT DATA FROM ALL COMPOSITIONS")
    print("=" * 70)

    compositions = [
        {
            "name": "webern_pointillism",
            "title": "Webern Pointillism",
            "extract_fn": extract_webern_events,
            "seed": None,
        },
        {
            "name": "berg_lyrical",
            "title": "Berg Lyrical",
            "extract_fn": extract_berg_lyrical_events,
            "seed": None,
        },
        {
            "name": "berg_extended",
            "title": "Berg Extended (7 min)",
            "extract_fn": extract_berg_extended_events,
            "seed": 42,
        },
        {
            "name": "cage_ambient",
            "title": "Cage Ambient",
            "extract_fn": extract_cage_ambient_events,
            "seed": 99,
        },
        {
            "name": "bells_bergman",
            "title": "Bells with Bergman Clock",
            "extract_fn": extract_bells_bergman_events,
            "seed": 42,
        },
        {
            "name": "bells_pizz",
            "title": "Bells Pizzicato",
            "extract_fn": extract_bells_pizz_events,
            "seed": 77,
        },
        {
            "name": "bells_gentle",
            "title": "Gentle Bells",
            "extract_fn": extract_bells_gentle_events,
            "seed": 55,
        },
        {
            "name": "tubular_low",
            "title": "Tubular Low",
            "extract_fn": extract_tubular_low_events,
            "seed": 31,
        },
    ]

    results = []

    for comp in compositions:
        print(f"\n{comp['title']}...")
        try:
            # Extract events
            events, duration, audio_file = comp["extract_fn"](comp["seed"])

            # Prepare JSON structure
            data = {
                "title": comp["title"],
                "duration": duration,
                "seed": comp["seed"],
                "audio_file": audio_file,
                "n_events": len(events),
                "events": events
            }

            # Save JSON
            json_filename = f"events_{comp['name']}.json"
            json_path = f"/sessions/sweet-nice-volta/mnt/generator/{json_filename}"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Saved: {json_filename} ({len(events)} events)")

            # Render score
            png_filename = f"{comp['name']}_score.png"
            png_path = f"/sessions/sweet-nice-volta/mnt/generator/{png_filename}"
            render_score_png(comp["name"], events, duration, png_path)

            results.append({
                "composition": comp["name"],
                "title": comp["title"],
                "events_file": json_filename,
                "score_file": png_filename,
                "n_events": len(events),
                "duration": duration,
                "seed": comp["seed"],
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    total_events = sum(r["n_events"] for r in results)
    total_duration = sum(r["duration"] for r in results)

    for r in results:
        print(f"{r['title']:30s}  {r['n_events']:4d} events  {r['duration']:7.1f}s  {r['events_file']}")

    print("-" * 70)
    print(f"{'TOTAL':30s}  {total_events:4d} events  {total_duration:7.1f}s")
    print("=" * 70)

    # Also save a manifest
    manifest = {
        "compositions": results,
        "total_events": total_events,
        "total_duration": total_duration,
        "generated": "2026-02-10",
        "description": "Event data extracted from all CloudAutomat compositions for animated score player"
    }

    manifest_path = "/sessions/sweet-nice-volta/mnt/generator/events_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: events_manifest.json")
    print("All data ready for animated score player!")


if __name__ == "__main__":
    main()
