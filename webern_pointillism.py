"""
webern_pointillism.py — Anton Webern-inspired pointillist synthesis

Characteristics modeled from Webern's late works (Op.21-31):
  - Derived rows: 12-tone row built from symmetric trichord cells
  - Pointillism: isolated tones separated by silence
  - Wide register leaps: 3-5 octave range, rarely adjacent
  - Klangfarbenmelodie: each note gets unique timbral character
  - Extreme brevity: 45-60 seconds
  - Dynamic whisper: ppp to mp, nothing loud
  - Sparse: ~35-50 discrete sonic events total
  - Silence ratio: ~60% silence, 40% sound
  - Palindromic/mirror structures

Synthesis engine: organic modeling with per-partial envelopes,
attack transients, pitch micro-drift, delayed vibrato, filtered
noise components, and simple convolution reverb.

Usage:
    python webern_pointillism.py [--seed N] [--output filename.wav]
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import random
import argparse

# --- Config ---
SAMPLE_RATE = 44100
DURATION = 50  # seconds — Webern-scale brevity
BASE_FREQ = 261.63  # C4 middle C — central reference

# --- Row construction ---
# Trichord: [0, 1, 4] (tight chromatic cell + major third leap)
TRICHORD = [0, 1, 4]

def derive_row(trichord):
    """Build a 12-tone row from a trichord via Webern's derivation method."""
    used = set(trichord)
    row = list(trichord)

    ri = [(5 - t) % 12 for t in reversed(trichord)]
    row.extend(ri)
    used.update(ri)

    remaining = [p for p in range(12) if p not in used]
    r = [8, remaining[0], remaining[1]] if len(remaining) >= 2 else remaining
    row.extend(r[:3])
    used.update(r[:3])

    remaining = [p for p in range(12) if p not in used]
    row.extend(remaining)

    return np.array(row[:12])

def row_inversion(row):
    return (2 * row[0] - row) % 12

def row_retrograde(row):
    return row[::-1]


# =====================================================================
# TIMBRE PALETTE — now models instrument *behavior*, not just spectrum
# =====================================================================
# Each timbre defines:
#   harmonics:        base partial amplitudes [fund, 2nd, 3rd, ...]
#   partial_decay:    how much faster upper partials decay (multiplier per partial number)
#   attack:           attack time in seconds
#   noise_type:       "breath", "bow", "click", "none" — filtered noise on attack
#   noise_amount:     how much noise (0-1)
#   noise_bandwidth:  center freq for bandpass noise filter (Hz), or "wide"
#   vibrato_delay:    seconds before vibrato onset (0 = immediate)
#   vibrato_rate:     Hz
#   vibrato_depth:    cents
#   drift:            max pitch drift in cents over note lifetime
#   undertone:        amplitude of subharmonic (freq/2), 0 = none
#   decay_shape:      "exp" or "sustained"

TIMBRES = [
    {
        "name": "cello_pont",  # sul ponticello — glassy, harmonic-rich
        "harmonics": [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08],
        "partial_decay": 1.4,   # upper partials die 1.4x faster per number
        "attack": 0.06,
        "noise_type": "bow", "noise_amount": 0.12, "noise_bandwidth": 3000,
        "vibrato_delay": 0.3, "vibrato_rate": 4.5, "vibrato_depth": 12,
        "drift": 5, "undertone": 0.0, "decay_shape": "sustained"
    },
    {
        "name": "cello_tasto",  # sul tasto — dark, muffled
        "harmonics": [1.0, 0.3, 0.08, 0.02],
        "partial_decay": 2.0,
        "attack": 0.12,
        "noise_type": "bow", "noise_amount": 0.08, "noise_bandwidth": 1200,
        "vibrato_delay": 0.5, "vibrato_rate": 4.2, "vibrato_depth": 15,
        "drift": 8, "undertone": 0.03, "decay_shape": "sustained"
    },
    {
        "name": "flute_breathy",
        "harmonics": [1.0, 0.12, 0.04],
        "partial_decay": 1.8,
        "attack": 0.04,
        "noise_type": "breath", "noise_amount": 0.18, "noise_bandwidth": 5000,
        "vibrato_delay": 0.2, "vibrato_rate": 5.0, "vibrato_depth": 8,
        "drift": 3, "undertone": 0.0, "decay_shape": "sustained"
    },
    {
        "name": "clarinet_chalumeau",  # low register — hollow, woody
        "harmonics": [1.0, 0.02, 0.6, 0.01, 0.3, 0.01, 0.1],  # odd partials dominate
        "partial_decay": 1.3,
        "attack": 0.025,
        "noise_type": "breath", "noise_amount": 0.06, "noise_bandwidth": 2200,
        "vibrato_delay": 0.4, "vibrato_rate": 4.8, "vibrato_depth": 6,
        "drift": 4, "undertone": 0.0, "decay_shape": "sustained"
    },
    {
        "name": "bell_struck",
        "harmonics": [1.0, 0.65, 0.4, 0.2, 0.12, 0.06, 0.03, 0.015],
        "partial_decay": 0.8,   # bell partials ring almost equally long
        "attack": 0.002,
        "noise_type": "click", "noise_amount": 0.15, "noise_bandwidth": "wide",
        "vibrato_delay": 99, "vibrato_rate": 0, "vibrato_depth": 0,  # no vibrato
        "drift": 1, "undertone": 0.04, "decay_shape": "exp"
    },
    {
        "name": "glass_harmonica",
        "harmonics": [1.0, 0.0, 0.0, 0.35, 0.0, 0.18],
        "partial_decay": 0.9,
        "attack": 0.08,
        "noise_type": "breath", "noise_amount": 0.04, "noise_bandwidth": 6000,
        "vibrato_delay": 0.1, "vibrato_rate": 5.5, "vibrato_depth": 4,
        "drift": 2, "undertone": 0.02, "decay_shape": "exp"
    },
    {
        "name": "pizzicato",
        "harmonics": [1.0, 0.55, 0.3, 0.18, 0.1, 0.05],
        "partial_decay": 1.6,
        "attack": 0.001,
        "noise_type": "click", "noise_amount": 0.2, "noise_bandwidth": "wide",
        "vibrato_delay": 99, "vibrato_rate": 0, "vibrato_depth": 0,
        "drift": 6, "undertone": 0.0, "decay_shape": "exp"
    },
    {
        "name": "oboe_pp",
        "harmonics": [1.0, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04],
        "partial_decay": 1.2,
        "attack": 0.015,
        "noise_type": "breath", "noise_amount": 0.05, "noise_bandwidth": 3500,
        "vibrato_delay": 0.25, "vibrato_rate": 5.2, "vibrato_depth": 10,
        "drift": 4, "undertone": 0.0, "decay_shape": "sustained"
    },
]


def freq_from_pitch_class(pc, octave):
    """Convert pitch class (0-11) and octave to frequency"""
    return BASE_FREQ * (2 ** ((pc - 0) / 12 + (octave - 4)))


def generate_noise(n_samples, noise_type, bandwidth, rng):
    """Generate shaped noise for attack transients.

    Real instruments produce noise at the onset:
      - bow:    broadband friction filtered around string resonance
      - breath: turbulent airflow, bandpass around embouchure
      - click:  very short broadband impulse (pluck, key, hammer)
    """
    raw = rng.randn(n_samples)

    if noise_type == "click":
        # Very short impulse — just the first ~2ms
        click_len = int(0.002 * SAMPLE_RATE)
        envelope = np.zeros(n_samples)
        envelope[:min(click_len, n_samples)] = np.exp(-np.linspace(0, 8, min(click_len, n_samples)))
        return raw * envelope

    if bandwidth == "wide":
        return raw

    # Bandpass filter the noise around the instrument's characteristic frequency
    center = bandwidth
    width = center * 0.6  # 60% bandwidth
    low = max(center - width / 2, 50)
    high = min(center + width / 2, SAMPLE_RATE / 2 - 100)

    # Simple 2nd order bandpass via biquad
    w0 = 2 * np.pi * center / SAMPLE_RATE
    Q = center / max(width, 1)
    alpha = np.sin(w0) / (2 * Q)

    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]

    filtered = lfilter(b, a, raw)
    # Normalize
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak
    return filtered


def pointillist_tone(t, start, pitch_class, octave, duration, amplitude, timbre, rng):
    """Generate a single tone with organic synthesis modeling.

    Key differences from simple additive:
      1. Per-partial envelopes: upper harmonics decay faster
      2. Attack transient: filtered noise burst at onset
      3. Pitch micro-drift: slow random walk
      4. Delayed vibrato: ramps in like a real player
      5. Slight inharmonicity: partials aren't perfect ratios
      6. Optional undertone (subharmonic)
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)
    freq = freq_from_pitch_class(pitch_class, octave)

    if freq < 30 or freq > 9000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = np.sum(mask)
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)  # 0..1 over note life

    # --- 1. PITCH MICRO-DRIFT ---
    # Random walk in cents, bounded by timbre's drift parameter
    max_drift_cents = timbre["drift"]
    # Generate smooth random drift via cumulative sum of small steps
    n_drift_points = max(int(duration * 20), 4)  # 20 control points per second
    drift_walk = np.cumsum(rng.randn(n_drift_points) * 0.3)
    # Bound it
    drift_walk = np.clip(drift_walk, -max_drift_cents, max_drift_cents)
    # Interpolate to sample rate
    drift_cents = np.interp(
        np.linspace(0, 1, n_active),
        np.linspace(0, 1, n_drift_points),
        drift_walk
    )
    # Convert cents to frequency multiplier
    freq_drift = freq * (2 ** (drift_cents / 1200))

    # --- 2. DELAYED VIBRATO ---
    vib_delay = timbre["vibrato_delay"]
    vib_rate = timbre["vibrato_rate"]
    vib_depth_cents = timbre["vibrato_depth"]

    if vib_delay < duration and vib_rate > 0:
        # Vibrato ramps in after delay
        vib_onset = np.clip((t_local - vib_delay) / 0.3, 0, 1)  # 0.3s ramp
        # Slight rate variation (real players aren't metronomic)
        rate_wobble = 1.0 + 0.05 * np.sin(2 * np.pi * 0.3 * t_local)
        vibrato = vib_onset * vib_depth_cents * np.sin(
            2 * np.pi * vib_rate * rate_wobble * t_local
        )
        freq_with_vib = freq_drift * (2 ** (vibrato / 1200))
    else:
        freq_with_vib = freq_drift

    # --- 3. PER-PARTIAL SYNTHESIS with individual envelopes ---
    signal = np.zeros(n_active)
    base_harmonics = timbre["harmonics"]
    partial_decay_rate = timbre["partial_decay"]

    for h_num, h_amp in enumerate(base_harmonics, 1):
        if h_amp < 0.005:
            continue

        # Slight inharmonicity — increases with partial number
        # Real strings: f_n = n * f0 * sqrt(1 + B*n^2) where B is stiffness
        stiffness = rng.uniform(0.0001, 0.0004)
        inharmonic_ratio = h_num * np.sqrt(1 + stiffness * h_num * h_num)

        # Phase: accumulate instantaneous frequency for FM-correct synthesis
        inst_freq = freq_with_vib * inharmonic_ratio
        phase = np.cumsum(2 * np.pi * inst_freq / SAMPLE_RATE)
        # Random initial phase (avoids all partials starting aligned)
        phase += rng.uniform(0, 2 * np.pi)

        partial_signal = h_amp * np.sin(phase)

        # Per-partial envelope: upper harmonics decay faster
        if timbre["decay_shape"] == "exp":
            # Base decay rate, accelerated for upper partials
            decay_rate = 3.0 + (h_num - 1) * partial_decay_rate
            partial_env = np.exp(-decay_rate * t_norm)
        else:
            # Sustained shape, but upper partials still thin out
            base_sustain = np.maximum(0, 1.0 - t_norm * 0.3)
            upper_fade = np.exp(-(h_num - 1) * partial_decay_rate * 0.5 * t_norm)
            partial_env = base_sustain * upper_fade

        partial_signal *= partial_env
        signal += partial_signal

    # Normalize by sum of harmonic weights
    total_weight = sum(h for h in base_harmonics if h >= 0.005)
    if total_weight > 0:
        signal /= total_weight

    # --- 4. UNDERTONE (subharmonic) ---
    undertone_amp = timbre["undertone"]
    if undertone_amp > 0:
        sub_phase = np.cumsum(2 * np.pi * (freq_with_vib * 0.5) / SAMPLE_RATE)
        sub_phase += rng.uniform(0, 2 * np.pi)
        # Subharmonic fades in slowly and has its own envelope
        sub_env = np.minimum(t_norm * 3, 1.0) * np.exp(-1.5 * t_norm)
        signal += undertone_amp * np.sin(sub_phase) * sub_env

    # --- 5. ATTACK ENVELOPE ---
    attack_time = timbre["attack"]
    attack_samples = max(int(attack_time * SAMPLE_RATE), 1)

    envelope = np.ones(n_active)

    # Attack: slightly exponential rise (not linear — linear sounds mechanical)
    if attack_samples < n_active:
        attack_curve = np.linspace(0, 1, min(attack_samples, n_active))
        # Exponential shape: fast initial rise, slowing near peak
        attack_curve = 1.0 - np.exp(-3.5 * attack_curve)
        attack_curve /= attack_curve[-1]  # normalize to reach 1.0
        envelope[:len(attack_curve)] = attack_curve

    # Decay/release (overall)
    if timbre["decay_shape"] == "exp":
        envelope *= np.exp(-2.5 * t_norm)
    else:
        # Sustained: gradual taper with soft ending
        envelope *= np.maximum(0, 1.0 - t_norm * 0.25)
        envelope *= np.exp(-0.8 * t_norm)

    # Anti-click fadeout (last 15ms)
    fadeout = min(int(0.015 * SAMPLE_RATE), n_active)
    if fadeout > 0:
        envelope[-fadeout:] *= np.linspace(1, 0, fadeout)

    # --- 6. ATTACK NOISE TRANSIENT ---
    noise_amount = timbre["noise_amount"]
    if noise_amount > 0:
        noise = generate_noise(n_active, timbre["noise_type"], timbre["noise_bandwidth"], rng)

        # Noise envelope: concentrated at attack, fading quickly
        if timbre["noise_type"] == "click":
            noise_env = np.exp(-np.linspace(0, 30, n_active))
        elif timbre["noise_type"] == "bow":
            # Bow noise persists throughout but peaks at attack
            noise_env = 0.3 + 0.7 * np.exp(-np.linspace(0, 6, n_active))
            # Slight bow pressure variation
            noise_env *= 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t_local)
        else:
            # Breath noise: attack-heavy with residual
            noise_env = 0.15 + 0.85 * np.exp(-np.linspace(0, 8, n_active))

        signal += noise * noise_env * noise_amount

    # Apply overall envelope and amplitude
    voice[mask] = signal * envelope * amplitude
    return voice


def simple_reverb(audio, decay=0.4, delays_ms=None, sample_rate=44100):
    """Simple multi-tap delay reverb.

    Not convolution (too heavy for this context) but multiple
    feedback delay lines at prime-number-ratio intervals create
    a reasonable sense of space. Good enough for pointillist material
    where the silence between events matters as much as the sound.
    """
    if delays_ms is None:
        # Prime-ish delay times for diffuse reflections
        delays_ms = [23, 37, 53, 71, 97, 131, 173, 229]

    wet = np.zeros_like(audio)

    for i, delay_ms in enumerate(delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000)
        tap_gain = decay * (0.85 ** i)  # each successive tap quieter

        delayed = np.zeros_like(audio)
        if delay_samples < len(audio):
            delayed[delay_samples:] = audio[:-delay_samples] * tap_gain

        # Alternate stereo would be nice but we're mono; just sum
        wet += delayed

    # Mix: mostly dry for this style
    return audio * 0.82 + wet * 0.18


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
    phrases = [
        {"start": 1.0,  "end": 16.0, "row_form": 0, "density": "sparse"},
        {"start": 18.0, "end": 34.0, "row_form": 1, "density": "medium"},
        {"start": 36.0, "end": 48.0, "row_form": 2, "density": "sparse"},
    ]

    events = []

    for phrase in phrases:
        p_start = phrase["start"]
        p_end = phrase["end"]
        active_row = row_forms[phrase["row_form"]]

        if phrase["density"] == "sparse":
            n_events = rng.randint(8, 10)
        else:
            n_events = rng.randint(12, 16)

        event_times = sorted([rng.uniform(p_start, p_end - 1.0) for _ in range(n_events)])

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

            octave = rng.choice([2, 3, 4, 5, 6])

            if rng.random() < 0.15:
                dur = rng.uniform(1.2, 2.5)
            elif rng.random() < 0.3:
                dur = rng.uniform(0.5, 1.0)
            else:
                dur = rng.uniform(0.08, 0.4)

            if rng.random() < 0.1:
                amp = rng.uniform(0.12, 0.18)
            elif rng.random() < 0.3:
                amp = rng.uniform(0.06, 0.12)
            else:
                amp = rng.uniform(0.02, 0.06)

            timbre = rng.choice(TIMBRES)

            events.append({
                "time": event_time,
                "pc": int(pc),
                "octave": octave,
                "duration": dur,
                "amplitude": amp,
                "timbre": timbre
            })

    # Occasional dyads
    n_dyads = rng.randint(2, 4)
    for _ in range(n_dyads):
        if events:
            source = rng.choice(events)
            companion_pc = (source["pc"] + rng.choice([3, 4, 7, 8, 11])) % 12
            companion_oct = source["octave"] + rng.choice([-2, -1, 1, 2])
            companion_oct = max(2, min(6, companion_oct))

            events.append({
                "time": source["time"] + rng.uniform(-0.02, 0.02),
                "pc": companion_pc,
                "octave": companion_oct,
                "duration": source["duration"] * rng.uniform(0.5, 1.5),
                "amplitude": source["amplitude"] * rng.uniform(0.5, 0.9),
                "timbre": rng.choice(TIMBRES)
            })

    # Render all events
    for event in events:
        audio += pointillist_tone(
            t, event["time"], event["pc"], event["octave"],
            event["duration"], event["amplitude"], event["timbre"],
            np_rng
        )

    # Apply reverb — gives events a physical space
    audio = simple_reverb(audio, decay=0.35, sample_rate=SAMPLE_RATE)

    # Global envelope
    fade_in = np.minimum(t / 0.5, 1.0)
    fade_out = np.minimum((DURATION - t) / 2.0, 1.0)
    audio *= fade_in * fade_out

    # Normalize with headroom
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak * 1.2)

    # Very gentle saturation
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

    wavfile.write(args.output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nGenerated: {args.output}")
    print(f"  Total events: {len(events)}")

    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(f"\n  Event map:")
    for e in sorted(events, key=lambda x: x["time"]):
        name = pitch_names[e["pc"]]
        dyn = "ppp" if e["amplitude"] < 0.06 else ("pp" if e["amplitude"] < 0.12 else "mp")
        print(f"    {e['time']:6.2f}s  {name}{e['octave']}  {e['duration']:.2f}s  {dyn:>3s}  [{e['timbre']['name']}]")


if __name__ == "__main__":
    main()
