"""
bells_score.py — Generate graphical score for the bells/Bergman piece
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import random

from bells_bergman import generate_bells_piece, freq_from_pc, DURATION

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_from_pc(pc, octave):
    return pc + (octave + 1) * 12


def midi_to_name(midi_note):
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES[pc]}{octave}"


def render_bells_score(events, output="bells_score.png"):
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    fig.patch.set_facecolor('#0f0a12')
    ax.set_facecolor('#0f0a12')

    PROFILE_COLORS = {
        'glockenspiel':  '#66ccff',   # bright ice blue
        'celesta':       '#cc99ff',   # soft violet
        'tubular_bell':  '#ffaa44',   # warm amber
        'church_bell':   '#ff6644',   # deep copper
        'wind_chime':    '#aaffcc',   # pale green sparkle
    }

    PROFILE_SHAPES = {
        'glockenspiel':  'D',    # diamond
        'celesta':       'o',    # circle
        'tubular_bell':  's',    # square
        'church_bell':   'p',    # pentagon
        'wind_chime':    '*',    # star
    }

    # --- Section backgrounds ---
    sections = [
        (0, 15, 'Clock Alone', '#1a1020'),
        (15, 40, 'First Bells', '#1a1525'),
        (40, 75, 'Accumulation', '#1a1a2a'),
        (75, 100, 'Full Peal', '#201a2a'),
        (100, 120, 'Dissolution', '#1a1020'),
    ]
    for s_start, s_end, s_name, s_color in sections:
        ax.axvspan(s_start, s_end, alpha=0.4, color=s_color, zorder=0)
        ax.text((s_start + s_end) / 2, 98, s_name,
               ha='center', fontsize=9, color='#555566',
               fontstyle='italic', zorder=1)

    # --- Clock tick marks along the bottom ---
    clock_bpm = 63
    period = 60.0 / clock_bpm
    n_beats = int(DURATION / period) + 1
    for i in range(n_beats):
        beat_time = i * period
        if beat_time >= DURATION:
            break
        is_tick = (i % 2 == 0)
        color = '#554433' if is_tick else '#443322'
        height = 1.5 if is_tick else 1.0
        ax.plot([beat_time, beat_time], [23, 23 + height],
               color=color, linewidth=0.5, alpha=0.6, zorder=1)

    # Label the clock
    ax.text(7.5, 21, 'TICK—TOCK—TICK—TOCK', ha='center', fontsize=7,
           color='#554433', fontstyle='italic', family='monospace', zorder=2)

    # --- Bell events ---
    for ev in events:
        midi = midi_from_pc(ev['pc'], ev['octave'])
        color = PROFILE_COLORS.get(ev['profile'], '#ffffff')
        marker = PROFILE_SHAPES.get(ev['profile'], 'o')

        # Size based on amplitude
        size = ev['amp'] * 2000

        ax.scatter(
            ev['time'], midi,
            s=size, c=color, marker=marker,
            alpha=0.8, zorder=5,
            edgecolors='white', linewidths=0.2
        )

        # Ring decay trail — fading horizontal line
        ring_len = ev['dur']
        n_trail = 20
        trail_x = np.linspace(ev['time'], ev['time'] + ring_len, n_trail)
        trail_alpha = np.linspace(0.5, 0.0, n_trail)
        for j in range(n_trail - 1):
            ax.plot(
                [trail_x[j], trail_x[j + 1]],
                [midi, midi],
                color=color, alpha=float(trail_alpha[j]),
                linewidth=1.5, zorder=3
            )

    # --- Axes ---
    ax.set_xlim(-1, DURATION + 1)

    all_midis = [midi_from_pc(e['pc'], e['octave']) for e in events]
    if all_midis:
        y_lo = min(all_midis) - 4
        y_hi = max(all_midis) + 4
    else:
        y_lo, y_hi = 24, 96
    ax.set_ylim(y_lo, y_hi)

    # Y ticks at octave C's
    y_ticks = list(range(24, 109, 12))
    y_ticks = [y for y in y_ticks if y_lo <= y <= y_hi]
    y_labels = [midi_to_name(n) for n in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, color='#aaaaaa')

    for yt in y_ticks:
        ax.axhline(y=yt, color='#222233', linewidth=0.4, alpha=0.5, zorder=0)

    x_ticks = list(range(0, DURATION + 1, 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}s" for x in x_ticks], fontsize=9, color='#aaaaaa')

    for xt in x_ticks:
        ax.axvline(x=xt, color='#222233', linewidth=0.3, alpha=0.4, zorder=0)

    ax.set_xlabel("Time (seconds)", fontsize=12, color='#cccccc', labelpad=10)
    ax.set_ylabel("Pitch", fontsize=12, color='#cccccc', labelpad=10)

    ax.set_title(
        "LAYERED BELLS WITH BERGMAN CLOCK — Graphical Score",
        fontsize=18, fontweight='bold', color='#ffffff', pad=20
    )

    ax.text(
        DURATION / 2, y_hi + 2,
        "CloudAutomat Labs  •  seed: 42  •  120 seconds  •  "
        "D Dorian → A Mixolydian → G Major → D Minor",
        ha='center', fontsize=9, color='#666688', style='italic'
    )

    # Legend
    legend_items = [
        ('◆', '#66ccff', 'Glockenspiel'),
        ('●', '#cc99ff', 'Celesta'),
        ('■', '#ffaa44', 'Tubular Bell'),
        ('⬠', '#ff6644', 'Church Bell'),
        ('★', '#aaffcc', 'Wind Chime'),
        ('|', '#554433', 'Clock tick-tock'),
    ]

    legend_y = y_hi - 2
    legend_x = DURATION - 16
    for i, (sym, color, label) in enumerate(legend_items):
        y_pos = legend_y - i * 2.0
        ax.text(legend_x, y_pos, sym, fontsize=12, color=color,
                fontweight='bold', ha='right', va='center', zorder=10)
        ax.text(legend_x + 0.5, y_pos, label, fontsize=8, color='#999999',
                ha='left', va='center', zorder=10)

    for spine in ax.spines.values():
        spine.set_color('#333344')
        spine.set_linewidth(0.5)
    ax.tick_params(colors='#666688')

    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight', facecolor='#0f0a12')
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    print("Extracting events...")
    _, events = generate_bells_piece(seed=42)
    print(f"  {len(events)} bell events")
    print("Rendering score...")
    render_bells_score(events, output="bells_score.png")
