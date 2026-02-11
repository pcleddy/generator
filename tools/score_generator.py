"""
score_generator.py — Generate musical scores from our compositions

Outputs both:
  1. GRAPHICAL SCORE (Cage-style) — time vs pitch with shapes/colors
     per event type. This is historically appropriate: Cage's own
     Concert for Piano and Orchestra (1958) used graphical notation.
  2. TRADITIONAL NOTATION — via music21 if available, or LilyPond markup

The graphical score is the main event here. It captures what a
traditional staff really can't: prepared piano preparations, tone
clusters, vocal screams, ambient drift — all as visual art.

Usage:
    python score_generator.py [--seed 99] [--output score.png]
"""

import numpy as np
import random
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

# Import our engines
from webern_pointillism import (
    SAMPLE_RATE, TIMBRES as INST_TIMBRES, freq_from_pitch_class
)
from berg_vocal import VOICE_TIMBRES

BASE_FREQ = 261.63
DURATION = 150

# Note names for labeling
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pitch_to_midi(pitch_class, octave):
    """Convert pitch class + octave to MIDI note number."""
    return pitch_class + (octave + 1) * 12


def midi_to_freq(midi_note):
    """MIDI note to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def midi_to_name(midi_note):
    """MIDI note to readable name."""
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES[pc]}{octave}"


# =====================================================================
# EVENT EXTRACTION — re-run the generation logic, capture events
# =====================================================================

def extract_cage_events(seed=99):
    """Re-run cage_ambient event generation, capture structured data.

    Returns list of event dicts with:
      type, start, duration, pitches, amplitude, preparation, category
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    events = []

    # --- Ambient bed events ---
    twelve_tone = np.array([0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10])
    inv = (12 - twelve_tone) % 12

    # Primary row voices
    for cycle in range(3):
        for i, pitch in enumerate(twelve_tone):
            start = cycle * 50 + i * 4.2
            if start + 8 > DURATION:
                break
            timbre = random.choice([INST_TIMBRES[0], INST_TIMBRES[1],
                                    INST_TIMBRES[5]])
            dur = rng.uniform(6, 12)
            amp = rng.uniform(0.04, 0.07)
            events.append({
                'type': 'ambient_primary',
                'category': 'ambient',
                'start': start,
                'duration': dur,
                'pitches': [pitch_to_midi(int(pitch), 3)],
                'amplitude': amp,
                'timbre': timbre.get('name', 'string'),
                'preparation': None,
                'row_position': i,
            })

    # Inverted row, higher register
    for cycle in range(2):
        for i, pitch in enumerate(inv):
            start = 5 + cycle * 55 + i * 4.8
            if start + 8 > DURATION:
                break
            timbre = random.choice([INST_TIMBRES[2], INST_TIMBRES[7]])
            dur = rng.uniform(5, 10)
            amp = rng.uniform(0.02, 0.045)
            events.append({
                'type': 'ambient_inverted',
                'category': 'ambient',
                'start': start,
                'duration': dur,
                'pitches': [pitch_to_midi(int(pitch), 4)],
                'amplitude': amp,
                'timbre': timbre.get('name', 'wind'),
                'preparation': None,
                'row_position': i,
            })

    # Sub-bass drones
    for start in [0, 40, 80, 120]:
        events.append({
            'type': 'ambient_drone',
            'category': 'ambient',
            'start': start,
            'duration': 35,
            'pitches': [pitch_to_midi(0, 2)],  # C2
            'amplitude': 0.03,
            'timbre': 'cello_tasto',
            'preparation': None,
        })

    # --- Cage interruptions ---
    # Replicate the chance logic from cage_ambient.py exactly
    n_interruptions = rng.randint(5, 8)
    available_window = DURATION - 25
    interruption_times = sorted(rng.sample(
        [x * 0.5 for x in range(30, int(available_window * 2))],
        min(n_interruptions, int(available_window / 12))
    ))

    # Enforce minimum spacing
    filtered_times = [interruption_times[0]]
    for it in interruption_times[1:]:
        if it - filtered_times[-1] >= 12:
            filtered_times.append(it)
    interruption_times = filtered_times

    for i, int_time in enumerate(interruption_times):
        int_type = rng.choice([
            "cluster_bolt", "cluster_screw", "single_slap",
            "cascade", "cluster_rubber", "vocal_scream"
        ])

        if int_type.startswith("cluster"):
            prep = int_type.split("_")[1]
            center = rng.randint(0, 11)
            n_notes = rng.randint(8, 15)
            spread = rng.uniform(8, 18)
            dur = rng.uniform(1.5, 4.0)
            amp = rng.uniform(0.20, 0.35)

            # Generate individual cluster notes
            cluster_pitches = []
            cluster_oct = rng.choice([2, 3, 4])
            for j in range(n_notes):
                offset = np_rng.uniform(-spread / 2, spread / 2)
                note_pc = int((center + offset) % 12)
                note_oct = cluster_oct + int(offset // 12)
                note_oct = max(1, min(7, note_oct))
                cluster_pitches.append(pitch_to_midi(note_pc, note_oct))

            events.append({
                'type': 'cluster',
                'category': 'interruption',
                'start': int_time,
                'duration': dur,
                'pitches': cluster_pitches,
                'amplitude': amp,
                'preparation': prep,
                'int_index': i,
            })

        elif int_type == "single_slap":
            pc = rng.randint(0, 11)
            oct = rng.choice([2, 3])
            dur = rng.uniform(2, 5)
            amp = rng.uniform(0.25, 0.40)
            prep = rng.choice(["bolt", "screw"])
            events.append({
                'type': 'single_slap',
                'category': 'interruption',
                'start': int_time,
                'duration': dur,
                'pitches': [pitch_to_midi(pc, oct)],
                'amplitude': amp,
                'preparation': prep,
                'int_index': i,
            })

        elif int_type == "cascade":
            n_strikes = rng.randint(6, 12)
            cascade_pitches = []
            cascade_times = []
            for j in range(n_strikes):
                strike_time = int_time + j * rng.uniform(0.05, 0.15)
                pc = (j * rng.choice([1, 2, 3])) % 12
                oct = rng.choice([2, 3, 4, 5])
                dur = rng.uniform(0.8, 2.5)
                amp = rng.uniform(0.10, 0.25)
                prep = rng.choice(["bolt", "screw", "rubber"])
                cascade_pitches.append(pitch_to_midi(pc, oct))
                cascade_times.append(strike_time)

            events.append({
                'type': 'cascade',
                'category': 'interruption',
                'start': int_time,
                'duration': max(cascade_times) - int_time + dur,
                'pitches': cascade_pitches,
                'cascade_times': cascade_times,
                'amplitude': amp,
                'preparation': 'mixed',
                'int_index': i,
            })

        elif int_type == "vocal_scream":
            voice_t = rng.choice(VOICE_TIMBRES)
            oct = voice_t["octave_range"][1]
            pc = rng.randint(0, 11)
            dur = rng.uniform(1.5, 3.0)
            amp = rng.uniform(0.18, 0.30)
            events.append({
                'type': 'vocal_scream',
                'category': 'interruption',
                'start': int_time,
                'duration': dur,
                'pitches': [pitch_to_midi(pc, oct)],
                'amplitude': amp,
                'preparation': None,
                'voice_type': voice_t.get('name', 'voice'),
                'int_index': i,
            })

    return events


# =====================================================================
# GRAPHICAL SCORE — the main output
# =====================================================================

def render_graphical_score(events, output="score_graphical.png"):
    """Render a Cage-style graphical score.

    X axis = time (seconds)
    Y axis = pitch (MIDI note number, labeled with note names)
    Shapes/colors encode event type and dynamics.
    """
    fig, ax = plt.subplots(1, 1, figsize=(36, 14))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    # Color palette
    COLORS = {
        'ambient_primary': '#1a5276',     # deep blue
        'ambient_inverted': '#1a6b5a',    # deep teal
        'ambient_drone': '#1c1c3a',       # near-black indigo
        'cluster': '#cc3300',             # angry red
        'single_slap': '#ff6600',         # orange
        'cascade': '#ff3366',             # hot pink
        'vocal_scream': '#ffcc00',        # screaming yellow
    }

    PREP_MARKERS = {
        'bolt': 'x',      # metallic cross
        'screw': '*',     # jangling star
        'rubber': 's',    # solid square (thuddy)
        'mixed': 'D',     # diamond
    }

    # --- Draw ambient events as horizontal translucent bars ---
    for ev in events:
        if ev['category'] != 'ambient':
            continue

        color = COLORS.get(ev['type'], '#333355')

        for midi_note in ev['pitches']:
            width = ev['duration']
            alpha = min(ev['amplitude'] * 8, 0.6)
            height = 1.2

            rect = FancyBboxPatch(
                (ev['start'], midi_note - height / 2),
                width, height,
                boxstyle="round,pad=0.15",
                facecolor=color, edgecolor='none',
                alpha=alpha, zorder=1
            )
            ax.add_patch(rect)

    # --- Draw interruptions ---
    for ev in events:
        if ev['category'] != 'interruption':
            continue

        color = COLORS.get(ev['type'], '#ff0000')

        if ev['type'] == 'cluster':
            # Draw cluster as a vertical splash of markers
            for midi_note in ev['pitches']:
                marker = PREP_MARKERS.get(ev.get('preparation', 'bolt'), 'o')
                size = ev['amplitude'] * 300
                ax.scatter(
                    ev['start'] + np.random.uniform(0, 0.03),
                    midi_note,
                    s=size, c=color, marker=marker,
                    alpha=0.85, zorder=5,
                    edgecolors='white', linewidths=0.3
                )

            # Bracket showing cluster span
            pitches = ev['pitches']
            if len(pitches) > 1:
                y_lo, y_hi = min(pitches), max(pitches)
                ax.plot(
                    [ev['start'] - 0.3, ev['start'] - 0.3],
                    [y_lo - 0.5, y_hi + 0.5],
                    color=color, linewidth=2.5, alpha=0.7, zorder=4
                )
                # Duration line
                ax.plot(
                    [ev['start'], ev['start'] + ev['duration']],
                    [y_hi + 1, y_hi + 1],
                    color=color, linewidth=1.0, alpha=0.4,
                    linestyle='--', zorder=3
                )

        elif ev['type'] == 'single_slap':
            midi_note = ev['pitches'][0]
            marker = PREP_MARKERS.get(ev.get('preparation', 'bolt'), 'o')
            ax.scatter(
                ev['start'], midi_note,
                s=ev['amplitude'] * 600, c=color, marker=marker,
                alpha=0.9, zorder=5,
                edgecolors='white', linewidths=0.5
            )
            # Decay trail
            trail_x = np.linspace(ev['start'], ev['start'] + ev['duration'], 30)
            trail_y = np.full_like(trail_x, midi_note)
            trail_alpha = np.linspace(0.7, 0.0, 30)
            for j in range(len(trail_x) - 1):
                ax.plot(
                    [trail_x[j], trail_x[j + 1]],
                    [trail_y[j], trail_y[j + 1]],
                    color=color, alpha=float(trail_alpha[j]),
                    linewidth=2.0, zorder=4
                )

        elif ev['type'] == 'cascade':
            # Draw cascade as connected diagonal strikes
            cascade_times = ev.get('cascade_times', [ev['start']])
            for j, (ct, midi_note) in enumerate(zip(cascade_times, ev['pitches'])):
                size = ev['amplitude'] * 250
                ax.scatter(
                    ct, midi_note,
                    s=size, c=color, marker='v',
                    alpha=0.85, zorder=5,
                    edgecolors='white', linewidths=0.3
                )
                # Connect to next strike
                if j < len(cascade_times) - 1:
                    ax.plot(
                        [ct, cascade_times[j + 1]],
                        [midi_note, ev['pitches'][j + 1]],
                        color=color, linewidth=1.5, alpha=0.5,
                        linestyle='-', zorder=4
                    )

        elif ev['type'] == 'vocal_scream':
            midi_note = ev['pitches'][0]
            # Scream: wavy line emanating from a point
            ax.scatter(
                ev['start'], midi_note,
                s=400, c=color, marker='o',
                alpha=0.9, zorder=6,
                edgecolors='#ff0000', linewidths=1.5
            )
            # Vibrato waves
            wave_t = np.linspace(0, ev['duration'], 100)
            wave_y = midi_note + 2.0 * np.sin(2 * np.pi * 5.5 * wave_t) * \
                     np.exp(-wave_t / ev['duration'] * 1.5)
            wave_alpha = np.linspace(0.8, 0.0, 100)
            for j in range(len(wave_t) - 1):
                ax.plot(
                    [ev['start'] + wave_t[j], ev['start'] + wave_t[j + 1]],
                    [wave_y[j], wave_y[j + 1]],
                    color=color, alpha=float(wave_alpha[j]),
                    linewidth=2.0, zorder=5
                )
            # Label
            ax.annotate(
                'SCREAM', (ev['start'] + 0.3, midi_note + 3.5),
                color=color, fontsize=7, fontweight='bold',
                fontstyle='italic', alpha=0.8, zorder=7
            )

    # --- Ducking regions (silence after interruptions) ---
    interruptions = [ev for ev in events if ev['category'] == 'interruption']
    for ev in interruptions:
        duck_start = ev['start'] + 0.3
        duck_end = ev['start'] + ev['duration'] + 3.0
        ax.axvspan(duck_start, duck_end, alpha=0.06, color='white', zorder=0)

    # --- Axes and labels ---
    ax.set_xlim(-1, DURATION + 1)

    # Y axis: MIDI note range
    all_pitches = []
    for ev in events:
        all_pitches.extend(ev['pitches'])
    if all_pitches:
        y_lo = min(all_pitches) - 3
        y_hi = max(all_pitches) + 3
    else:
        y_lo, y_hi = 24, 96
    ax.set_ylim(y_lo, y_hi)

    # Y tick labels = note names every octave C
    y_ticks = list(range(24, 97, 12))  # C1 through C7
    y_labels = [midi_to_name(n) for n in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, color='#aaaaaa')

    # Horizontal grid at each octave
    for yt in y_ticks:
        ax.axhline(y=yt, color='#222244', linewidth=0.5, alpha=0.5, zorder=0)

    # X ticks every 10 seconds
    x_ticks = list(range(0, DURATION + 1, 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}s" for x in x_ticks], fontsize=9, color='#aaaaaa')

    # Vertical grid every 10s
    for xt in x_ticks:
        ax.axvline(x=xt, color='#222244', linewidth=0.3, alpha=0.4, zorder=0)

    ax.set_xlabel("Time (seconds)", fontsize=12, color='#cccccc', labelpad=10)
    ax.set_ylabel("Pitch", fontsize=12, color='#cccccc', labelpad=10)

    # Title
    ax.set_title(
        "AMBIENT DRIFT WITH CAGE INTERRUPTIONS — Graphical Score",
        fontsize=18, fontweight='bold', color='#ffffff',
        pad=20
    )

    # Subtitle
    ax.text(
        DURATION / 2, y_hi + 1.5,
        "CloudAutomat Labs / generator project  •  seed: 99  •  150 seconds  •  prepared piano + ambient + voice",
        ha='center', fontsize=9, color='#666688', style='italic'
    )

    # --- Legend ---
    legend_items = [
        ('■', '#1a5276', 'Ambient (primary row)'),
        ('■', '#1a6b5a', 'Ambient (inverted row)'),
        ('■', '#1c1c3a', 'Sub-bass drone'),
        ('×', '#cc3300', 'Cluster (bolt prep)'),
        ('★', '#cc3300', 'Cluster (screw prep)'),
        ('●', '#ff6600', 'Single slap'),
        ('▼', '#ff3366', 'Cascade'),
        ('●', '#ffcc00', 'Vocal scream'),
        ('░', '#ffffff', 'Ducking / silence'),
    ]

    legend_y = y_hi - 2
    legend_x = DURATION - 22
    for i, (sym, color, label) in enumerate(legend_items):
        y_pos = legend_y - i * 1.8
        ax.text(legend_x, y_pos, sym, fontsize=12, color=color,
                fontweight='bold', ha='right', va='center', zorder=10)
        ax.text(legend_x + 0.5, y_pos, label, fontsize=8, color='#999999',
                ha='left', va='center', zorder=10)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#333355')
        spine.set_linewidth(0.5)
    ax.tick_params(colors='#666688')

    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close()
    print(f"Saved graphical score: {output}")
    return output


# =====================================================================
# TRADITIONAL NOTATION — event list / text score
# =====================================================================

def render_text_score(events, output="score_text.txt"):
    """Generate a readable text score with all events notated.

    This is the 'performer's guide' — tells you what to do and when.
    Like Cage's performance instructions for prepared piano:
    you need a text description alongside (or instead of) staff notation.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("AMBIENT DRIFT WITH CAGE INTERRUPTIONS")
    lines.append("Text Score / Performance Instructions")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Duration: 150 seconds")
    lines.append("Instrumentation: Prepared piano, synthesized instruments,")
    lines.append("                 vocal performer, ambient electronics")
    lines.append("")
    lines.append("PREPARATION TABLE (place objects on piano strings):")
    lines.append("-" * 50)

    # Collect all preparations needed
    preps_used = set()
    for ev in events:
        if ev.get('preparation'):
            preps_used.add(ev['preparation'])

    prep_instructions = {
        'bolt': "  BOLT: Large steel bolt laid across strings. Creates metallic,\n"
                "        buzzing timbre with many inharmonic partials.",
        'screw': "  SCREW: Wood screw inserted between strings. Produces\n"
                 "         jangling, bright, sustained rattling tone.",
        'rubber': "  RUBBER: Rubber eraser wedged between strings. Dark,\n"
                  "          thuddy, heavily damped — very fast decay.",
    }
    for p in sorted(preps_used):
        if p in prep_instructions:
            lines.append(prep_instructions[p])
    lines.append("")

    # Sort all events by time
    sorted_events = sorted(events, key=lambda e: e['start'])

    lines.append("=" * 72)
    lines.append("SCORE")
    lines.append("=" * 72)
    lines.append("")

    current_section = None

    for ev in sorted_events:
        time_str = f"[{ev['start']:6.1f}s]"

        # Section headers
        if ev['start'] < 15 and current_section != 'opening':
            lines.append("--- SECTION I: OPENING (ambient drift) ---\n")
            current_section = 'opening'
        elif 15 <= ev['start'] < 50 and current_section != 'development':
            lines.append("\n--- SECTION II: DEVELOPMENT ---\n")
            current_section = 'development'
        elif 50 <= ev['start'] < 100 and current_section != 'middle':
            lines.append("\n--- SECTION III: MIDDLE ---\n")
            current_section = 'middle'
        elif ev['start'] >= 100 and current_section != 'closing':
            lines.append("\n--- SECTION IV: CLOSING ---\n")
            current_section = 'closing'

        if ev['category'] == 'ambient':
            pitch_names = [midi_to_name(p) for p in ev['pitches']]
            dyn = 'ppp' if ev['amplitude'] < 0.03 else 'pp' if ev['amplitude'] < 0.05 else 'p'
            timbre = ev.get('timbre', '?')

            if ev['type'] == 'ambient_drone':
                lines.append(f"  {time_str}  DRONE: {pitch_names[0]}, "
                            f"dur={ev['duration']:.0f}s, {dyn} (cello tasto)")
            else:
                row_type = "P" if ev['type'] == 'ambient_primary' else "I"
                row_pos = ev.get('row_position', '?')
                lines.append(f"  {time_str}  {row_type}[{row_pos:2}]: {pitch_names[0]:4s} "
                            f"dur={ev['duration']:.1f}s  {dyn}  ({timbre})")

        elif ev['category'] == 'interruption':
            # INTERRUPTION — big, dramatic notation
            lines.append(f"\n  {'!'*40}")

            if ev['type'] == 'cluster':
                pitch_names = sorted(set(midi_to_name(p) for p in ev['pitches']))
                lo = midi_to_name(min(ev['pitches']))
                hi = midi_to_name(max(ev['pitches']))
                dyn = 'ff' if ev['amplitude'] < 0.30 else 'fff'
                lines.append(f"  {time_str}  *** CLUSTER ({ev['preparation']}) ***")
                lines.append(f"             {len(ev['pitches'])} notes: {lo} to {hi}")
                lines.append(f"             Duration: {ev['duration']:.1f}s   Dynamic: {dyn}")
                lines.append(f"             Preparation: {ev['preparation']}")
                lines.append(f"             Notes: {', '.join(pitch_names)}")

            elif ev['type'] == 'single_slap':
                pitch = midi_to_name(ev['pitches'][0])
                dyn = 'ff' if ev['amplitude'] < 0.30 else 'fff'
                lines.append(f"  {time_str}  *** PIANO SLAP ({ev['preparation']}) ***")
                lines.append(f"             Pitch: {pitch}")
                lines.append(f"             Duration: {ev['duration']:.1f}s   Dynamic: {dyn}")

            elif ev['type'] == 'cascade':
                lines.append(f"  {time_str}  *** CASCADE ***")
                lines.append(f"             {len(ev['pitches'])} rapid strikes:")
                for j, (ct, p) in enumerate(zip(
                    ev.get('cascade_times', []), ev['pitches']
                )):
                    lines.append(f"               Strike {j+1}: {midi_to_name(p)} "
                                f"at {ct:.2f}s")
                lines.append(f"             Total span: {ev['duration']:.1f}s")

            elif ev['type'] == 'vocal_scream':
                pitch = midi_to_name(ev['pitches'][0])
                lines.append(f"  {time_str}  *** VOCAL SCREAM ***")
                lines.append(f"             Pitch: {pitch}")
                lines.append(f"             Duration: {ev['duration']:.1f}s")
                lines.append(f"             Wide vibrato, open 'AH' vowel")
                lines.append(f"             Voice type: {ev.get('voice_type', '?')}")

            lines.append(f"  {'!'*40}\n")
            lines.append(f"  {'':>10}  [SILENCE — ambient ducks for ~3s]\n")

    # Summary statistics
    lines.append("\n" + "=" * 72)
    lines.append("STATISTICS")
    lines.append("=" * 72)

    n_ambient = sum(1 for e in events if e['category'] == 'ambient')
    n_interrupt = sum(1 for e in events if e['category'] == 'interruption')
    types_count = {}
    for e in events:
        types_count[e['type']] = types_count.get(e['type'], 0) + 1

    lines.append(f"  Total events: {len(events)}")
    lines.append(f"  Ambient events: {n_ambient}")
    lines.append(f"  Interruptions: {n_interrupt}")
    lines.append(f"  Event types:")
    for t, c in sorted(types_count.items()):
        lines.append(f"    {t}: {c}")

    # Pitch range
    all_pitches = [p for e in events for p in e['pitches']]
    lines.append(f"  Pitch range: {midi_to_name(min(all_pitches))} — "
                f"{midi_to_name(max(all_pitches))}")

    lines.append("")
    lines.append("NOTE: This score is generated from the same random seed (99)")
    lines.append("that produced the audio. Re-running cage_ambient.py --seed 99")
    lines.append("will produce the identical performance.")
    lines.append("")
    lines.append("CloudAutomat Labs — generator project")
    lines.append("=" * 72)

    text = "\n".join(lines)
    with open(output, 'w') as f:
        f.write(text)
    print(f"Saved text score: {output}")
    return output


# =====================================================================
# LILYPOND OUTPUT (bonus — if user has lilypond installed)
# =====================================================================

def render_lilypond(events, output="score.ly"):
    """Generate LilyPond markup for traditional notation.

    Note: prepared piano clusters and vocal screams will be
    approximated — some things just can't be staff-notated.
    That's why Cage invented graphical notation.
    """
    ly_lines = []
    ly_lines.append('\\version "2.24.0"')
    ly_lines.append('\\header {')
    ly_lines.append('  title = "Ambient Drift with Cage Interruptions"')
    ly_lines.append('  subtitle = "Algorithmic composition — seed 99"')
    ly_lines.append('  composer = "CloudAutomat Labs / generator"')
    ly_lines.append('}')
    ly_lines.append('')
    ly_lines.append('\\paper {')
    ly_lines.append('  #(set-paper-size "a3" \'landscape)')
    ly_lines.append('}')
    ly_lines.append('')

    # We'll create a simplified piano reduction
    # Group events into 10-second bins for sanity
    ly_lines.append('% NOTE: This is a simplified reduction.')
    ly_lines.append('% The full graphical score (score_graphical.png) is the')
    ly_lines.append('% authoritative notation for this piece.')
    ly_lines.append('')

    # Just output the interruption events as a timeline
    ly_lines.append('\\markup {')
    ly_lines.append('  \\column {')
    ly_lines.append('    \\line { "INTERRUPTION TIMELINE:" }')
    ly_lines.append('    \\vspace #1')

    interruptions = [e for e in events if e['category'] == 'interruption']
    for ev in interruptions:
        pitches = ', '.join(midi_to_name(p) for p in ev['pitches'][:5])
        if len(ev['pitches']) > 5:
            pitches += f' (+{len(ev["pitches"])-5} more)'
        prep = ev.get('preparation', 'none')
        ly_lines.append(f'    \\line {{ "{ev["start"]:.1f}s: '
                       f'{ev["type"]} [{prep}] — {pitches}" }}')

    ly_lines.append('  }')
    ly_lines.append('}')

    text = "\n".join(ly_lines)
    with open(output, 'w') as f:
        f.write(text)
    print(f"Saved LilyPond source: {output}")
    return output


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate musical scores from cage_ambient composition"
    )
    parser.add_argument("--seed", type=int, default=99, help="Random seed (must match audio)")
    parser.add_argument("--output", type=str, default="score_graphical.png",
                        help="Graphical score output filename")
    args = parser.parse_args()

    print(f"Extracting events from cage_ambient (seed={args.seed})...")
    events = extract_cage_events(seed=args.seed)
    print(f"  Found {len(events)} events")
    print(f"    Ambient: {sum(1 for e in events if e['category'] == 'ambient')}")
    print(f"    Interruptions: {sum(1 for e in events if e['category'] == 'interruption')}")

    print("\nRendering graphical score...")
    render_graphical_score(events, output=args.output)

    print("\nRendering text score...")
    render_text_score(events, output="score_text.txt")

    print("\nRendering LilyPond source...")
    render_lilypond(events, output="score.ly")

    print("\nDone! Three score formats generated.")


if __name__ == "__main__":
    main()
