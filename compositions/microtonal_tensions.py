"""
microtonal_tensions.py — Microtonal tension and resolution.

A 30-second study in using microtonal drift to create and release
emotional tension. Inspired by the spectral school (Grisey, Haas):
intervals start pure, drift into the roughness zone (~20-50 cents
off from just), hold the tension, then resolve — either back to
pure or forward to the next consonant interval.

Three arcs:
  1. Pure fifth → wolf fifth → resolution to unison
  2. Octave → slowly detuned → snaps to pure major 6th
  3. Two voices converging from a minor 3rd into a devastating
     unison through microtonal space

Each arc uses the continuous space between semitones for emotion,
not chord changes. The dissonance is in the beating, the roughness,
the almost-but-not-quite.

Direct synthesis — bypasses Renderer for exact frequency control.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile
import subprocess

from synthesis_engine.config import SAMPLE_RATE
from synthesis_engine.seed_manager import SeedManager
from synthesis_engine.synthesis.plucked import karplus_strong
from synthesis_engine.synthesis.reverb import simple_reverb
from synthesis_engine.profiles.plucked import PLUCKED_PROFILES

NYLON = PLUCKED_PROFILES["guitar_nylon"]
CELLO = PLUCKED_PROFILES["plucked_cello"]


def cents_to_ratio(cents):
    return 2 ** (cents / 1200.0)


def render_note(t, time, freq, dur, amp, profile, rng):
    """Render a single K-S note at exact frequency."""
    return karplus_strong(t, time, freq, dur, min(amp, 0.6), profile, rng)


def generate():
    rng = SeedManager(113)

    duration = 35.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    audio = np.zeros(len(t))

    # ══════════════════════════════════════════════════════════
    # ARC 1: Pure fifth → wolf fifth → resolves to unison
    # t=0 to t=10
    # ══════════════════════════════════════════════════════════
    # D3 (146.83) + A3 (220.00) = pure fifth (702 cents)
    # The A slowly drifts flat — through the "wolf" zone at ~680 cents
    # (22 cents flat, maximum beating) — then continues down to D4 octave
    # equivalence... no, resolves by collapsing to unison on D3.

    root = 146.83  # D3

    print("  ARC 1: Fifth → wolf → unison (0-10s)")

    # Root note: steady D3, three strikes
    for strike_t in [0.3, 3.5, 7.0]:
        # Root drifts down very slightly (-8 cents total) for unease
        root_drift = -8.0 * (strike_t / 10.0)
        f = root * cents_to_ratio(root_drift)
        audio += render_note(t, strike_t, f, 3.0, 0.50, NYLON, rng)

    # Upper voice: starts as pure fifth, drifts flat into wolf, resolves
    fifth_strikes = [
        # (time, cents_above_root, amp) — the emotional journey
        (0.8,   702,  0.45),   # pure fifth — consonant, warm
        (2.5,   690,  0.42),   # 12 cents flat — first unease, slight beating
        (4.2,   672,  0.44),   # 30 cents flat — deep in the wolf zone, ROUGH
        (5.8,   655,  0.40),   # 47 cents flat — agonizing, maximum tension
        (7.5,   640,  0.38),   # passing through — still tense
        (8.8,   610,  0.35),   # approaching tritone territory
        (9.6,   5,    0.50),   # RESOLUTION: snaps to near-unison (5 cents off)
    ]

    for strike_t, cents_above, amp in fifth_strikes:
        root_drift = -8.0 * (strike_t / 10.0)
        f = root * cents_to_ratio(root_drift + cents_above)
        audio += render_note(t, strike_t, f, 2.5, amp, NYLON, rng)
        beating = ""
        if 660 < cents_above < 700 and cents_above != 702:
            beat_hz = abs(root * cents_to_ratio(702) - root * cents_to_ratio(cents_above))
            beating = f"  ~{beat_hz:.1f}Hz beating"
        print(f"    t={strike_t:.1f}  fifth={cents_above:+.0f}¢{beating}")

    # ══════════════════════════════════════════════════════════
    # ARC 2: Octave → microtonal sourness → pure major 6th
    # t=10 to t=20
    # ══════════════════════════════════════════════════════════
    # G3 (196.00) + G4 (392.00) = pure octave
    # Upper voice sags flat — octave becomes a "dirty" near-octave
    # (beating, sourness) — then the ear reinterprets as it lands
    # on a pure major 6th (884 cents).

    g3 = 196.00
    print("\n  ARC 2: Octave → sour → major 6th (10-20s)")

    # Lower voice: G3, steady with tiny humanization
    for strike_t in [10.5, 14.0, 17.5]:
        f = g3 * cents_to_ratio(rng.uniform(-2, 2))
        audio += render_note(t, strike_t, f, 3.2, 0.48, CELLO, rng)

    octave_strikes = [
        (11.0,  1200, 0.44),   # pure octave — open, clear
        (12.8,  1175, 0.42),   # 25 cents flat — queasy
        (14.5,  1140, 0.44),   # 60 cents flat — very sour, ambiguous
        (16.0,  1070, 0.40),   # falling further — where are we going?
        (17.5,  980,  0.38),   # approaching minor 7th territory
        (19.0,  884,  0.50),   # RESOLUTION: pure major 6th (5:3 ratio)
    ]

    for strike_t, cents_above, amp in octave_strikes:
        f = g3 * cents_to_ratio(cents_above)
        audio += render_note(t, strike_t, f, 2.8, amp, CELLO, rng)
        print(f"    t={strike_t:.1f}  interval={cents_above:.0f}¢ "
              f"({cents_above/100:.1f} semitones)")

    # ══════════════════════════════════════════════════════════
    # ARC 3: Two voices converging from minor 3rd → unison
    # t=20 to t=30
    # ══════════════════════════════════════════════════════════
    # E3 high and Db3 low, 300 cents apart (minor 3rd).
    # They inch toward each other through microtonal space —
    # maximum roughness when they're ~30-50 cents apart —
    # then land together on D3. The emotional payoff.

    e3 = 164.81
    db3 = 138.59  # Db3 — 300 cents below E3
    d3 = 146.83   # D3 — the meeting point

    print("\n  ARC 3: Minor 3rd converges → unison (20-30s)")

    convergence = [
        # (time, upper_freq, lower_freq, label)
        (20.5,  e3,                     db3,                    "minor 3rd — 300¢ apart"),
        (22.0,  e3 * cents_to_ratio(-40),  db3 * cents_to_ratio(30),  "230¢ apart — closing"),
        (23.5,  e3 * cents_to_ratio(-80),  db3 * cents_to_ratio(65),  "185¢ apart — narrowing"),
        (25.0,  e3 * cents_to_ratio(-120), db3 * cents_to_ratio(100), "80¢ apart — tension building"),
        (26.5,  e3 * cents_to_ratio(-145), db3 * cents_to_ratio(120), "35¢ apart — MAXIMUM ROUGHNESS"),
        (28.0,  d3 * cents_to_ratio(3),    d3 * cents_to_ratio(-3),   "6¢ apart — almost there"),
        (29.5,  d3,                     d3,                     "UNISON — resolution"),
    ]

    for strike_t, f_upper, f_lower, label in convergence:
        gap = 1200 * np.log2(f_upper / f_lower) if f_upper != f_lower else 0
        # Upper voice: nylon (brighter)
        amp_u = 0.48 if gap > 50 else 0.52  # louder at resolution
        audio += render_note(t, strike_t, f_upper, 2.5, amp_u, NYLON, rng)
        # Lower voice: cello (darker) — slightly delayed for humanity
        amp_l = 0.42 if gap > 50 else 0.52
        audio += render_note(t, strike_t + 0.08, f_lower, 2.5, amp_l, CELLO, rng)
        if gap > 1:
            beat_hz = abs(f_upper - f_lower)
            print(f"    t={strike_t:.1f}  gap={gap:.0f}¢  beating={beat_hz:.1f}Hz  {label}")
        else:
            print(f"    t={strike_t:.1f}  {label}")

    # ══════════════════════════════════════════════════════════
    # Mix and master
    # ══════════════════════════════════════════════════════════

    # Deep reverb — let the beating and resolution breathe
    audio = simple_reverb(audio, preset="deep", sample_rate=SAMPLE_RATE)

    # Fade in and out
    fade_in = int(0.5 * SAMPLE_RATE)
    audio[:fade_in] *= np.linspace(0, 1, fade_in)
    fade_out = int(3.0 * SAMPLE_RATE)
    audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.92

    return audio, duration


def main():
    print("Microtonal Tensions — Three arcs of tension & resolution")
    print("  Using microtonal drift through the roughness zone")
    print()

    audio, dur = generate()

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "microtonal_tensions.wav")
    mp3_path = os.path.join(out_dir, "audio", "microtonal_tensions.mp3")

    audio_16 = np.int16(audio * 32767)
    wavfile.write(wav_path, SAMPLE_RATE, audio_16)
    print(f"\n  Written: {wav_path}")

    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-codec:a', 'libmp3lame', '-b:a', '192k', mp3_path],
        capture_output=True
    )
    print(f"  Written: {mp3_path}")


if __name__ == "__main__":
    main()
