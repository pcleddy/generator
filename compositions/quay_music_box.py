#!/usr/bin/env python3
"""
quay_music_box.py — "The Dormitory of Dust"
=============================================
A Quay Brothers–inspired music box piece.

Synthesis concepts:
  - Music box tine: steel comb plucked by cylinder pins. More harmonic than bells
    but with characteristic metallic overtone at ~5.4× and fast attack from pin-pluck.
  - Winding-down: spring tension modeled as exponential decay → tempo decelerates,
    pitch sags, inter-note silence grows.
  - Mechanical textures: gear clicks, spring creaks, key-winding ratchet.
  - Distant bells: faint tubular bells from another room.
  - Room dust: very quiet broadband noise — the ambience of an old cabinet.

Structure (3:00):
  0–8s    Winding Up — ratchet clicks accelerate, spring tightens
  8–55s   The Melody — music box waltz in G major, steady tempo ~108 BPM
  55–90s  First Slowing — tempo decelerates to ~72 BPM, occasional missed tines
  90–100s Mechanical Stutter — gears catch, irregular clicks
  100–140s Final Playing — melody resumes slower, more fragmented, pitch sagging
  140–170s Winding Down — extreme deceleration, notes stretched, silences grow
  170–180s Stillness — room dust, one last tine

No samples, no DAW. Pure numpy/scipy.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import argparse
import os

SAMPLE_RATE = 44100
DURATION = 180  # 3 minutes

# ============================================================
# MUSIC BOX TINE PROFILE
# ============================================================
# Steel tines plucked by brass pins on a rotating cylinder.
# More harmonic than bells (tuned steel) but with metallic shimmer.
# The pin-pluck creates a very short, bright transient.

MUSIC_BOX_TINE = {
    "partials": [
        (1.0,   1.0,   4.0),    # fundamental — quick decay
        (2.0,   0.55,  3.5),    # 2nd harmonic
        (3.0,   0.25,  5.0),    # 3rd
        (4.0,   0.12,  6.0),    # 4th — fading
        (5.404, 0.35,  2.5),    # metallic overtone — THE music box sound, rings longer
        (6.0,   0.08,  7.0),    # high shimmer
    ],
    "strike_brightness": 8000,   # very bright pin pluck
    "strike_amount": 0.15,       # sharp but not harsh
    "ring_time_mult": 0.8,       # tines ring briefly
}

# Detuned / worn music box — some tines slightly off
MUSIC_BOX_WORN = {
    "partials": [
        (1.0,   1.0,   4.5),    # slightly faster decay (worn tine)
        (1.997, 0.50,  4.0),    # slightly flat 2nd
        (3.01,  0.22,  5.5),    # slightly sharp 3rd
        (4.0,   0.10,  6.5),
        (5.38,  0.30,  3.0),    # metallic overtone, shifted
        (6.1,   0.06,  8.0),
    ],
    "strike_brightness": 7000,
    "strike_amount": 0.12,
    "ring_time_mult": 0.7,
}

# ============================================================
# SCALES AND MELODY
# ============================================================

# G major scale for the waltz melody
# pc: G=7, A=9, B=11, C=0, D=2, E=4, F#=6
G_MAJOR = [7, 9, 11, 0, 2, 4, 6]

# A simple, slightly melancholy waltz melody (pitch classes + octave offsets)
# Each tuple: (scale_degree, octave_offset, duration_beats, accent)
# scale_degree 0-6 maps into G_MAJOR
WALTZ_MELODY = [
    # Phrase 1 — descending figure
    (4, 1, 1.0, 0.8),   # D5
    (3, 1, 0.5, 0.5),   # C5
    (2, 1, 0.5, 0.5),   # B4
    (1, 0, 1.0, 0.6),   # A4
    (0, 0, 2.0, 0.7),   # G4 — hold
    # Phrase 2 — ascending
    (0, 0, 1.0, 0.8),   # G4
    (1, 0, 0.5, 0.5),   # A4
    (2, 0, 0.5, 0.5),   # B4
    (3, 1, 1.0, 0.6),   # C5
    (4, 1, 2.0, 0.7),   # D5 — hold
    # Phrase 3 — peak and fall
    (5, 1, 1.0, 0.9),   # E5
    (4, 1, 1.0, 0.6),   # D5
    (2, 0, 1.0, 0.5),   # B4
    (0, 0, 3.0, 0.7),   # G4 — long hold
    # Phrase 4 — tender ending
    (1, 0, 1.0, 0.6),   # A4
    (0, 0, 0.5, 0.4),   # G4
    (6, -1, 0.5, 0.4),  # F#4
    (0, 0, 3.0, 0.8),   # G4 — resolve
]

# Bass notes for waltz (downbeats) — pedal tones
WALTZ_BASS = [
    (0, -1, 3.0, 0.35),  # G3
    (0, -1, 3.0, 0.30),  # G3
    (3, -1, 3.0, 0.30),  # C4 (actually octave -1 from base = C4)
    (4, -1, 3.0, 0.30),  # D4
    (0, -1, 3.0, 0.35),  # G3
    (5, -1, 3.0, 0.25),  # E4 (relative minor hint)
    (0, -1, 3.0, 0.35),  # G3
]

BASE_FREQ = 261.63  # C4


def freq_from_pc(pc, octave):
    """Convert pitch class + octave to frequency in Hz."""
    return BASE_FREQ * (2 ** ((pc - 0) / 12 + (octave - 4)))


# ============================================================
# SYNTHESIS: MUSIC BOX TINE
# ============================================================

def tine_strike(t, start, freq, duration, amplitude, profile, rng, pitch_sag=0.0):
    """
    Synthesize a single music box tine being plucked.

    pitch_sag: cents of downward pitch drift (models spring winding down).
    """
    sr = SAMPLE_RATE
    start_samp = int(start * sr)
    dur_samp = int(duration * sr)
    if start_samp >= len(t) or dur_samp <= 0:
        return np.zeros_like(t)

    end_samp = min(start_samp + dur_samp, len(t))
    n_samp = end_samp - start_samp
    local_t = np.arange(n_samp) / sr

    voice = np.zeros(len(t))

    # Pitch sag over note duration (spring losing tension)
    if pitch_sag > 0:
        sag_curve = np.linspace(0, -pitch_sag, n_samp)
        freq_curve = freq * (2 ** (sag_curve / 1200))
    else:
        freq_curve = freq

    # Synthesize each partial
    for ratio, amp, decay_rate in profile["partials"]:
        # Slight random detuning (manufacturing imperfection)
        detune = 1.0 + rng.uniform(-0.002, 0.002)
        partial_freq = freq_curve * ratio * detune

        # Phase
        if isinstance(partial_freq, np.ndarray):
            phase = np.cumsum(2 * np.pi * partial_freq / sr)
        else:
            phase = 2 * np.pi * partial_freq * local_t

        # Decay envelope — music box tines have a fast initial decay then ring
        actual_decay = decay_rate / profile["ring_time_mult"]
        env = np.exp(-actual_decay * local_t)

        # Add slight beating between close partials
        beat_freq = rng.uniform(0.5, 2.0)
        beat_depth = rng.uniform(0.0, 0.05)
        beat = 1.0 + beat_depth * np.sin(2 * np.pi * beat_freq * local_t)

        partial_audio = amp * env * beat * np.sin(phase)
        voice[start_samp:end_samp] += partial_audio[:n_samp]

    # Pin-pluck transient — very short, bright click
    pluck_dur = int(0.003 * sr)  # 3ms
    if pluck_dur > 0 and pluck_dur <= n_samp:
        pluck_noise = rng.randn(pluck_dur)
        # High-pass to get the bright "tick" of the pin
        pluck_env = np.exp(-np.linspace(0, 12, pluck_dur))
        pluck_noise *= pluck_env
        # Bandpass around brightness frequency
        nyq = sr / 2
        bright = min(profile["strike_brightness"], nyq * 0.9)
        low = max(bright * 0.5, 20) / nyq
        high = min(bright * 1.5, nyq * 0.95) / nyq
        if low < high and low > 0:
            b_bp, a_bp = butter(2, [low, high], btype='band')
            pluck_noise = lfilter(b_bp, a_bp, pluck_noise)
        voice[start_samp:start_samp + pluck_dur] += (
            pluck_noise * profile["strike_amount"] * amplitude
        )

    # Amplitude scaling
    voice[start_samp:end_samp] *= amplitude

    # Anti-click fadeout (10ms)
    fade_samp = min(int(0.010 * sr), n_samp)
    if fade_samp > 0:
        fade = np.linspace(1, 0, fade_samp)
        voice[end_samp - fade_samp:end_samp] *= fade

    return voice


# ============================================================
# MECHANICAL TEXTURES
# ============================================================

def make_gear_click(t, time_pos, amplitude, rng):
    """Single mechanical click — gear tooth or ratchet."""
    sr = SAMPLE_RATE
    start = int(time_pos * sr)
    click_dur = int(rng.uniform(0.001, 0.004) * sr)
    if start >= len(t) or click_dur <= 0:
        return np.zeros_like(t)

    end = min(start + click_dur, len(t))
    n = end - start
    click = rng.randn(n)
    click *= np.exp(-np.linspace(0, 20, n))

    # Bandpass for metallic quality
    nyq = sr / 2
    center = rng.uniform(2000, 6000)
    low = max(center * 0.5, 100) / nyq
    high = min(center * 1.5, nyq * 0.95) / nyq
    if low < high and low > 0 and n > 12:
        b, a = butter(2, [low, high], btype='band')
        click = lfilter(b, a, click)

    out = np.zeros_like(t)
    out[start:end] = click[:n] * amplitude
    return out


def make_spring_creak(t, start_time, duration, amplitude, rng):
    """Creaking spring / mechanical stress sound."""
    sr = SAMPLE_RATE
    s = int(start_time * sr)
    dur = int(duration * sr)
    if s >= len(t) or dur <= 0:
        return np.zeros_like(t)

    end = min(s + dur, len(t))
    n = end - s
    local_t = np.arange(n) / sr

    # Slowly sweeping filtered noise — like a hinge or spring
    sweep_freq = rng.uniform(300, 800)
    sweep_range = rng.uniform(100, 400)
    sweep_rate = rng.uniform(2, 8)

    center = sweep_freq + sweep_range * np.sin(2 * np.pi * sweep_rate * local_t)

    # Generate noise and filter in chunks
    chunk_size = int(0.01 * sr)
    creak = np.zeros(n)
    noise = rng.randn(n) * 0.5
    nyq = sr / 2

    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        cn = chunk_end - i
        freq = float(np.mean(center[i:chunk_end]))
        bw = freq * 0.3
        low = max((freq - bw), 50) / nyq
        high = min((freq + bw), nyq * 0.9) / nyq
        if low < high and low > 0 and cn > 12:
            b, a = butter(2, [low, high], btype='band')
            creak[i:chunk_end] = lfilter(b, a, noise[i:chunk_end])
        else:
            creak[i:chunk_end] = noise[i:chunk_end] * 0.1

    # Amplitude envelope — swell and fade
    env = np.sin(np.pi * local_t / (n / sr)) ** 0.5
    creak *= env

    out = np.zeros_like(t)
    out[s:end] = creak[:n] * amplitude
    return out


def make_winding_ratchet(t, start_time, end_time, clicks_per_sec_start,
                         clicks_per_sec_end, amplitude, rng):
    """Ratchet winding sound — accelerating or decelerating clicks."""
    sr = SAMPLE_RATE
    duration = end_time - start_time
    n_clicks = int(duration * (clicks_per_sec_start + clicks_per_sec_end) / 2)

    out = np.zeros_like(t)

    for i in range(n_clicks):
        # Interpolate click rate
        frac = i / max(n_clicks - 1, 1)
        rate = clicks_per_sec_start + frac * (clicks_per_sec_end - clicks_per_sec_start)
        elapsed = 0.0
        if rate > 0:
            # Accumulate time based on instantaneous rate
            elapsed = i / rate
        click_time = start_time + elapsed
        if click_time >= end_time:
            break

        # Alternating louder/softer (ratchet mechanism has two phases)
        click_amp = amplitude * (0.7 + 0.3 * (i % 2))
        out += make_gear_click(t, click_time, click_amp, rng)

    return out


def make_room_dust(t, amplitude, rng):
    """Very quiet broadband noise — dusty old cabinet ambience."""
    noise = rng.randn(len(t)) * amplitude

    # Low-pass to remove harshness — just a warm hiss
    nyq = SAMPLE_RATE / 2
    cutoff = 2000 / nyq
    b, a = butter(3, cutoff, btype='low')
    dust = lfilter(b, a, noise)

    # Very slow amplitude modulation — breathing of the room
    breath_rate = rng.uniform(0.05, 0.15)
    breath = 0.7 + 0.3 * np.sin(2 * np.pi * breath_rate * t)
    dust *= breath

    return dust


# ============================================================
# REVERB — small wooden box resonance
# ============================================================

def box_reverb(audio, decay=0.35, sample_rate=44100):
    """
    Small-box reverb — shorter delays than cathedral, more intimate.
    Models the wooden cabinet of a music box.
    """
    # Short prime-ratio delays for small enclosed space
    delays_ms = [7, 11, 17, 23, 31, 41, 53, 67]
    wet = np.zeros_like(audio)

    for i, delay_ms in enumerate(delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000)
        tap_gain = decay * (0.82 ** i)
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples] * tap_gain
        wet += delayed

    # More dry, less wet — intimate space
    return audio * 0.88 + wet * 0.12


# ============================================================
# SPRING TENSION MODEL
# ============================================================

def spring_tension(time_in_piece, sections):
    """
    Returns (tempo_multiplier, pitch_sag_cents) for a given time.
    Models a music box spring that winds down over the piece.
    """
    for section in sections:
        if section["start"] <= time_in_piece < section["end"]:
            frac = (time_in_piece - section["start"]) / (section["end"] - section["start"])
            tempo = section["tempo_start"] + frac * (section["tempo_end"] - section["tempo_start"])
            sag = section["sag_start"] + frac * (section["sag_end"] - section["sag_start"])
            return tempo, sag
    return 0.0, 100.0  # stopped


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_piece(seed=42):
    rng = np.random.RandomState(seed)
    np_rng = np.random.RandomState(seed + 1)
    t = np.linspace(0, DURATION, DURATION * SAMPLE_RATE, endpoint=False)
    audio = np.zeros_like(t)
    events = []

    # ---- Section timing and spring model ----
    SECTIONS = [
        # Winding up
        {"start": 0,   "end": 8,   "tempo_start": 0.0,   "tempo_end": 0.0,
         "sag_start": 0, "sag_end": 0},
        # Steady melody
        {"start": 8,   "end": 55,  "tempo_start": 108.0, "tempo_end": 106.0,
         "sag_start": 0, "sag_end": 5},
        # First slowing
        {"start": 55,  "end": 90,  "tempo_start": 106.0, "tempo_end": 68.0,
         "sag_start": 5, "sag_end": 25},
        # Mechanical stutter
        {"start": 90,  "end": 100, "tempo_start": 68.0,  "tempo_end": 55.0,
         "sag_start": 25, "sag_end": 35},
        # Final playing
        {"start": 100, "end": 140, "tempo_start": 55.0,  "tempo_end": 36.0,
         "sag_start": 35, "sag_end": 60},
        # Winding down
        {"start": 140, "end": 170, "tempo_start": 36.0,  "tempo_end": 8.0,
         "sag_start": 60, "sag_end": 120},
        # Stillness
        {"start": 170, "end": 180, "tempo_start": 0.0,   "tempo_end": 0.0,
         "sag_start": 120, "sag_end": 150},
    ]

    # ==== 1. ROOM DUST (entire piece) ====
    audio += make_room_dust(t, amplitude=0.008, rng=np_rng)

    # ==== 2. WINDING UP (0-8s) ====
    # Ratchet clicks accelerating as key is turned
    audio += make_winding_ratchet(t, 0.5, 7.5, 2, 12, 0.06, rng)

    # A spring creak as tension builds
    audio += make_spring_creak(t, 3.0, 2.0, 0.025, rng)
    audio += make_spring_creak(t, 6.0, 1.5, 0.03, rng)

    # ==== 3. THE MELODY — music box waltz ====
    # We'll step through the melody repeatedly, applying spring tension
    # to determine the actual tempo and pitch sag at each note.

    melody_time = 8.0  # Start of melody
    melody_cycle = 0
    max_cycles = 12  # Enough to fill the duration

    for cycle in range(max_cycles):
        # Melody voice
        for deg, oct_off, dur_beats, accent in WALTZ_MELODY:
            if melody_time >= 170:
                break

            tempo, sag = spring_tension(melody_time, SECTIONS)
            if tempo <= 0:
                melody_time += 0.5
                continue

            # Convert beats to seconds at current tempo
            beat_dur = 60.0 / tempo
            note_dur = dur_beats * beat_dur

            # Pitch
            pc = G_MAJOR[deg]
            octave = 5 + oct_off  # Base octave 5 for music box

            # Occasional missed tine (increases as spring winds down)
            miss_chance = max(0, (melody_time - 55) / 200)
            if rng.random() < miss_chance:
                melody_time += note_dur
                continue

            freq = freq_from_pc(pc, octave)

            # Choose worn or normal tine
            profile = MUSIC_BOX_WORN if rng.random() < 0.3 else MUSIC_BOX_TINE

            # Amplitude decreases slightly as spring winds down
            spring_amp = max(0.15, 1.0 - (melody_time - 8) / 300)
            amp = accent * 0.22 * spring_amp

            voice = tine_strike(t, melody_time, freq, note_dur * 2.5,
                                amp, profile, rng, pitch_sag=sag)
            audio += voice

            events.append({
                'time': melody_time,
                'pc': pc,
                'octave': octave,
                'duration': note_dur,
                'amplitude': amp,
                'type': 'music_box',
                'category': 'melody',
            })

            melody_time += note_dur

        if melody_time >= 170:
            break

    # ==== 4. BASS NOTES — lower register pedal tones ====
    bass_time = 8.0
    bass_idx = 0

    while bass_time < 165:
        tempo, sag = spring_tension(bass_time, SECTIONS)
        if tempo <= 0:
            bass_time += 1.0
            continue

        beat_dur = 60.0 / tempo
        bass_note = WALTZ_BASS[bass_idx % len(WALTZ_BASS)]
        deg, oct_off, dur_beats, accent = bass_note

        note_dur = dur_beats * beat_dur
        pc = G_MAJOR[deg]
        octave = 4 + oct_off  # One octave lower

        freq = freq_from_pc(pc, octave)
        spring_amp = max(0.1, 1.0 - (bass_time - 8) / 350)
        amp = accent * 0.14 * spring_amp

        voice = tine_strike(t, bass_time, freq, note_dur * 2.0,
                            amp, MUSIC_BOX_TINE, rng, pitch_sag=sag * 1.2)
        audio += voice

        events.append({
            'time': bass_time,
            'pc': pc,
            'octave': octave,
            'duration': note_dur,
            'amplitude': amp,
            'type': 'music_box',
            'category': 'bass',
        })

        bass_time += note_dur
        bass_idx += 1

    # ==== 5. MECHANICAL STUTTER (90-100s) ====
    # Irregular gear clicks — the mechanism catching
    for _ in range(25):
        click_time = rng.uniform(90, 100)
        click_amp = rng.uniform(0.02, 0.06)
        audio += make_gear_click(t, click_time, click_amp, rng)

    # Spring stress creaks
    audio += make_spring_creak(t, 91, 3.0, 0.035, rng)
    audio += make_spring_creak(t, 96, 2.5, 0.03, rng)

    # ==== 6. DISTANT BELLS — from another room ====
    # Very quiet, very reverberant, tubular bell partials
    DISTANT_BELL = {
        "partials": [
            (0.5,   0.20, 1.2),
            (1.0,   1.0,  1.5),
            (1.183, 0.60, 1.8),
            (1.506, 0.35, 2.0),
            (2.0,   0.45, 2.2),
            (2.514, 0.15, 3.0),
        ],
        "strike_brightness": 2000,
        "strike_amount": 0.04,
        "ring_time_mult": 3.5,   # very long ring — distant
    }

    # A few distant bell tolls
    bell_times = [35, 72, 115, 155]
    bell_pcs = [2, 7, 0, 2]  # D, G, C, D
    bell_octs = [3, 2, 3, 2]

    for bt, bpc, boct in zip(bell_times, bell_pcs, bell_octs):
        freq = freq_from_pc(bpc, boct)
        bell_voice = tine_strike(t, bt, freq, 12.0, 0.04, DISTANT_BELL, rng)
        audio += bell_voice
        events.append({
            'time': bt,
            'pc': bpc,
            'octave': boct,
            'duration': 12.0,
            'amplitude': 0.04,
            'type': 'tubular_bell',
            'category': 'distant_bell',
        })

    # ==== 7. SCATTERED MECHANICAL SOUNDS THROUGHOUT ====
    # Occasional gear clicks and creaks (the mechanism is never silent)
    for _ in range(40):
        click_time = rng.uniform(8, 170)
        click_amp = rng.uniform(0.005, 0.02)
        audio += make_gear_click(t, click_time, click_amp, rng)

    # Periodic spring creaks
    creak_times = [18, 32, 48, 65, 82, 108, 125, 148, 162]
    for ct in creak_times:
        dur = rng.uniform(0.8, 2.5)
        amp = rng.uniform(0.01, 0.025)
        audio += make_spring_creak(t, ct, dur, amp, rng)

    # ==== 8. ONE LAST TINE (175s) ====
    # The very last note — slow, pitched way down, ringing out alone
    last_freq = freq_from_pc(7, 4)  # G4 — home key
    last_voice = tine_strike(t, 175, last_freq, 5.0, 0.12,
                             MUSIC_BOX_WORN, rng, pitch_sag=80)
    audio += last_voice
    events.append({
        'time': 175,
        'pc': 7,
        'octave': 4,
        'duration': 5.0,
        'amplitude': 0.12,
        'type': 'music_box',
        'category': 'last_tine',
    })

    # ==== 9. WINDING-DOWN MECHANICAL SOUNDS (140-170s) ====
    # The spring making noise as it completely unwinds
    for i in range(8):
        creak_time = 142 + i * 3.5
        if creak_time < 170:
            audio += make_spring_creak(t, creak_time, rng.uniform(1.5, 3.0),
                                       rng.uniform(0.015, 0.035), rng)

    # Slow, heavy gear clicks as cylinder barely turns
    for i in range(15):
        click_time = 140 + rng.uniform(0, 30)
        if click_time < 170:
            audio += make_gear_click(t, click_time, rng.uniform(0.02, 0.05), rng)

    # ============================================================
    # MIX AND MASTER
    # ============================================================

    # Box reverb — small wooden enclosure
    audio = box_reverb(audio, decay=0.35, sample_rate=SAMPLE_RATE)

    # Gentle normalization — keep it intimate, not loud
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.72  # Leave headroom, keep it quiet

    # Very gentle high-shelf warmth (slight high-frequency rolloff)
    nyq = SAMPLE_RATE / 2
    b, a = butter(1, 12000 / nyq, btype='low')
    audio = lfilter(b, a, audio)

    return audio, events


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quay Brothers Music Box")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    args = parser.parse_args()

    print("Generating 'The Dormitory of Dust'...")
    print(f"  Seed: {args.seed}")
    print(f"  Duration: {DURATION}s")

    audio, events = generate_piece(seed=args.seed)

    output = args.output or f"quay_music_box_01.wav"
    wavfile.write(output, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"  Events: {len(events)}")
    print(f"  Output: {output}")
    print("Done.")
