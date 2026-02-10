"""
webern_pointillism.py — Anton Webern-inspired pointillist synthesis

Characteristics modeled from Webern's late works (Op.21-31):
  - Derived rows: 12-tone row built from symmetric trichord cells
  - Pointillism: isolated tones separated by silence
  - Wide register leaps: 3-5 octave range, rarely adjacent
  - Klangfarbenmelodie: each note gets unique timbral character
  - Extreme brevity: 45-60 seconds (Webern's Op.21 mvts are ~3-4 min;
    individual phrases are seconds long)
  - Dynamic whisper: ppp to mp, nothing loud
  - Sparse: ~35-50 discrete sonic events total
  - Silence ratio: ~60% silence, 40% sound
  - Palindromic/mirror structures (Webern loved symmetry)

Usage:
    python webern_pointillism.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
import random
import argparse

# --- Config ---
SAMPLE_RATE = 44100
DURATION = 50  # seconds — Webern-scale brevity
BASE_FREQ = 261.63  # C4 middle C — central reference

# --- Row construction ---
# Webern preferred derived rows: a single trichord generates the full row
# through transposition, inversion, retrograde. This mirrors his Op.24 approach.
# Trichord: [0, 1, 4] (a tight chromatic cell + major third leap)
TRICHORD = [0, 1, 4]

def derive_row(trichord):
    """Build a 12-tone row from a trichord via Webern's derivation method.

    Four forms of the trichord fill all 12 pitch classes:
      P  = original
      RI = retrograde inversion (transposed)
      R  = retrograde (transposed)
      I  = inversion (transposed)
    """
    used = set(trichord)
    row = list(trichord)

    # Retrograde inversion from pitch 5
    ri = [(5 - t) % 12 for t in reversed(trichord)]
    row.extend(ri)
    used.update(ri)

    # Fill remaining pitch classes maintaining interval character
    remaining = [p for p in range(12) if p not in used]
    # Retrograde from pitch 8
    r = [8, remaining[0], remaining[1]] if len(remaining) >= 2 else remaining
    row.extend(r[:3])
    used.update(r[:3])

    remaining = [p for p in range(12) if p not in used]
    row.extend(remaining)

    return np.array(row[:12])

def row_inversion(row):
    """Invert row around first pitch"""
    return (2 * row[0] - row) % 12

def row_retrograde(row):
    return row[::-1]

# --- Timbre palette (Klangfarbenmelodie) ---
# Each "color" is a harmonic recipe + attack character
TIMBRES = [
    {"name": "pure",      "harmonics": [1.0],                          "attack": 0.005, "decay_shape": "exp"},
    {"name": "hollow",    "harmonics": [1.0, 0.0, 0.3],                "attack": 0.01,  "decay_shape": "exp"},
    {"name": "bell",      "harmonics": [1.0, 0.6, 0.3, 0.15, 0.08],   "attack": 0.002, "decay_shape": "exp"},
    {"name": "whistle",   "harmonics": [0.3, 1.0, 0.1],                "attack": 0.04,  "decay_shape": "linear"},
    {"name": "glass",     "harmonics": [1.0, 0.0, 0.0, 0.4, 0.0, 0.2],"attack": 0.001, "decay_shape": "exp"},
    {"name": "breath",    "harmonics": [1.0, 0.15, 0.08, 0.04],        "attack": 0.08,  "decay_shape": "linear"},
    {"name": "pluck",     "harmonics": [1.0, 0.5, 0.25, 0.12, 0.06],  "attack": 0.001, "decay_shape": "exp"},
    {"name": "flute",     "harmonics": [1.0, 0.1, 0.05],               "attack": 0.03,  "decay_shape": "linear"},
]

def freq_from_pitch_class(pc, octave):
    """Convert pitch class (0-11) and octave to frequency"""
    return BASE_FREQ * (2 ** ((pc - 0) / 12 + (octave - 4)))

def pointillist_tone(t, start, pitch_class, octave, duration, amplitude, timbre, rng):
    """Generate a single isolated tone with unique timbral character.

    Each tone is a discrete sonic event — Webern's Klangfarbenmelodie.
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = freq_from_pitch_class(pitch_class, octave)

    # Clamp frequency to audible range
    if freq < 40 or freq > 8000:
        return voice

    mask = (t >= start) & (t < start + duration)
    if not np.any(mask):
        return voice

    t_local = t[mask] - start

    # Build harmonic content
    signal = np.zeros_like(t_local)
    for h_num, h_amp in enumerate(timbre["harmonics"], 1):
        # Slight inharmonicity for upper partials (like real instruments)
        inharmonicity = 1.0 + (h_num - 1) * rng.uniform(0, 0.003)
        signal += h_amp * np.sin(2 * np.pi * freq * h_num * inharmonicity * t_local)

    # Normalize harmonic mix
    signal /= max(sum(timbre["harmonics"]), 1.0)

    # Envelope: sharp or soft attack, always decaying
    attack_time = timbre["attack"]
    envelope = np.ones_like(t_local)

    # Attack phase
    attack_mask = t_local < attack_time
    if np.any(attack_mask):
        envelope[attack_mask] = t_local[attack_mask] / attack_time

    # Decay phase
    decay_start = attack_time
    decay_mask = t_local >= decay_start
    decay_local = (t_local[decay_mask] - decay_start) / (duration - decay_start + 1e-6)

    if timbre["decay_shape"] == "exp":
        # Exponential decay — bell-like, plucked
        envelope[decay_mask] = np.exp(-4.0 * decay_local)
    else:
        # Linear decay — breath-like, sustained
        envelope[decay_mask] = np.maximum(0, 1.0 - decay_local * 0.7)
        # Soft tail
        envelope[decay_mask] *= np.exp(-1.5 * decay_local)

    # Final fadeout to prevent clicks
    fadeout_samples = min(int(0.01 * SAMPLE_RATE), np.sum(mask))
    if fadeout_samples > 0:
        envelope[-fadeout_samples:] *= np.linspace(1, 0, fadeout_samples)

    voice[mask] = signal * envelope * amplitude
    return voice

def generate_webern_piece(seed=None):
    """Generate a pointillist piece using Webern-derived techniques."""

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
    else:
        rng = random.Random()
        np_rng = np.random.RandomState()

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros_like(t)

    # Build row and transformations
    row = derive_row(TRICHORD)
    inv = row_inversion(row)
    retro = row_retrograde(row)
    retro_inv = row_retrograde(inv)

    row_forms = [row, inv, retro, retro_inv]

    # --- Structure: three phrases with silence between ---
    # Webern's forms are tiny — think of three aphoristic gestures

    phrases = [
        {"start": 1.0,  "end": 16.0, "row_form": 0, "density": "sparse"},
        {"start": 18.0, "end": 34.0, "row_form": 1, "density": "medium"},
        {"start": 36.0, "end": 48.0, "row_form": 2, "density": "sparse"},  # palindrome: return
    ]

    events = []

    for phrase in phrases:
        p_start = phrase["start"]
        p_end = phrase["end"]
        p_duration = p_end - p_start
        active_row = row_forms[phrase["row_form"]]

        if phrase["density"] == "sparse":
            # 8-10 events, lots of silence
            n_events = rng.randint(8, 10)
        else:
            # 12-16 events, still sparse by normal standards
            n_events = rng.randint(12, 16)

        # Distribute events across phrase with irregular spacing
        # Webern: events cluster briefly then scatter
        event_times = sorted([rng.uniform(p_start, p_end - 1.0) for _ in range(n_events)])

        # Ensure minimum gap between events (silence is structural)
        min_gap = 0.3
        adjusted = [event_times[0]]
        for et in event_times[1:]:
            if et - adjusted[-1] >= min_gap:
                adjusted.append(et)
            else:
                pushed = adjusted[-1] + min_gap + rng.uniform(0.1, 0.5)
                if pushed < p_end - 0.5:
                    adjusted.append(pushed)
        event_times = adjusted

        for i, event_time in enumerate(event_times):
            pitch_idx = i % len(active_row)
            pc = active_row[pitch_idx]

            # Wide register distribution — Webern's signature
            # Avoid staying in same octave consecutively
            octave = rng.choice([2, 3, 4, 5, 6])

            # Very short durations — pointillistic
            # Occasional longer tone for contrast
            if rng.random() < 0.15:
                dur = rng.uniform(1.2, 2.5)   # rare sustained moment
            elif rng.random() < 0.3:
                dur = rng.uniform(0.5, 1.0)    # moderate
            else:
                dur = rng.uniform(0.08, 0.4)   # typical: a flicker

            # Dynamics: mostly very quiet, occasional mp
            if rng.random() < 0.1:
                amp = rng.uniform(0.12, 0.18)  # mp — the loudest it gets
            elif rng.random() < 0.3:
                amp = rng.uniform(0.06, 0.12)  # pp
            else:
                amp = rng.uniform(0.02, 0.06)  # ppp — most events

            # Klangfarbenmelodie: each note gets a different timbre
            timbre = rng.choice(TIMBRES)

            events.append({
                "time": event_time,
                "pc": int(pc),
                "octave": octave,
                "duration": dur,
                "amplitude": amp,
                "timbre": timbre
            })

    # --- Occasional dyads (two simultaneous pitches) ---
    # Webern used these sparingly for emphasis
    n_dyads = rng.randint(2, 4)
    for _ in range(n_dyads):
        # Pick a random existing event and add a companion
        if events:
            source = rng.choice(events)
            companion_pc = (source["pc"] + rng.choice([3, 4, 7, 8, 11])) % 12  # interval class
            companion_oct = source["octave"] + rng.choice([-2, -1, 1, 2])
            companion_oct = max(2, min(6, companion_oct))

            events.append({
                "time": source["time"] + rng.uniform(-0.02, 0.02),  # near-simultaneous
                "pc": companion_pc,
                "octave": companion_oct,
                "duration": source["duration"] * rng.uniform(0.5, 1.5),
                "amplitude": source["amplitude"] * rng.uniform(0.5, 0.9),
                "timbre": rng.choice(TIMBRES)
            })

    # --- Render all events ---
    for event in events:
        audio += pointillist_tone(
            t, event["time"], event["pc"], event["octave"],
            event["duration"], event["amplitude"], event["timbre"],
            np_rng
        )

    # --- Global envelope: fade in first 0.5s, fade out last 2s ---
    fade_in = np.minimum(t / 0.5, 1.0)
    fade_out = np.minimum((DURATION - t) / 2.0, 1.0)
    audio *= fade_in * fade_out

    # --- Normalize gently (preserve dynamics, leave headroom) ---
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.2)  # -1.6dB headroom

    # Soft saturation (very gentle — preserve transients)
    audio = np.tanh(audio * 1.05) / 1.05

    return audio, events

def main():
    parser = argparse.ArgumentParser(description="Webern-inspired pointillist synthesis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="webern_01.wav", help="Output filename")
    args = parser.parse_args()

    print(f"Generating Webern pointillist piece...")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Duration: {DURATION}s")
    print(f"  Row trichord: {TRICHORD}")

    audio, events = generate_webern_piece(seed=args.seed)

    # Write 16-bit WAV
    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")

    # Print event map
    print(f"\n  Event map:")
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for e in sorted(events, key=lambda x: x["time"]):
        name = pitch_names[e["pc"]]
        dyn = "ppp" if e["amplitude"] < 0.06 else ("pp" if e["amplitude"] < 0.12 else "mp")
        print(f"    {e['time']:6.2f}s  {name}{e['octave']}  {e['duration']:.2f}s  {dyn:>3s}  [{e['timbre']['name']}]")

if __name__ == "__main__":
    main()
