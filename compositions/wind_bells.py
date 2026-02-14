"""
wind_bells.py — 2 minutes of mid-range tubular bells stirred by wind.

Sparse, gentle strikes as if a breeze is nudging hanging tubes.
Irregular timing, soft dynamics, mid-range pitches (C4-G5).
Cathedral reverb to let the overtones bloom and overlap.
Each bell rings until it naturally dies — no hard cutoff.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile
import subprocess

from synthesis_engine.config import SAMPLE_RATE
from synthesis_engine.seed_manager import SeedManager
from synthesis_engine.synthesis.bell import bell_strike
from synthesis_engine.synthesis.reverb import simple_reverb

# Tubular bell profile — slightly warmer/longer than default
WIND_TUBE = {
    "name": "wind_tube",
    "category": "bell",
    "partials": [
        (0.5,   0.20,  1.2),   # sub-fundamental, slow decay
        (1.0,   1.0,   1.4),   # fundamental, long ring
        (1.183, 0.65,  1.6),
        (1.506, 0.40,  1.8),
        (2.0,   0.50,  2.0),
        (2.514, 0.15,  2.5),
        (3.011, 0.08,  3.0),
        (4.166, 0.03,  4.0),
    ],
    "strike_brightness": 2200,  # softer strike than standard tubular
    "strike_amount": 0.08,      # very gentle — wind, not mallet
    "ring_time_mult": 2.5,      # long ring
}

# Mid-range pitches: pentatonic-ish set for natural consonance
# when random wind-blown overlaps happen
FREQS = {
    "C4":  261.63,
    "D4":  293.66,
    "E4":  329.63,
    "G4":  392.00,
    "A4":  440.00,
    "C5":  523.25,
    "D5":  587.33,
    "G5":  783.99,
}


def generate():
    rng = SeedManager(42)

    content_time = 120.0  # 2 minutes of content
    duration = content_time + 12.0  # extra tail for final bell to ring out
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    audio = np.zeros(len(t))

    freq_list = list(FREQS.values())
    freq_names = list(FREQS.keys())

    # Wind-blown timing: irregular gaps, clustered and sparse
    current_time = 0.8
    strike_count = 0

    while current_time < content_time:
        # Pick a pitch — favor the middle of our range
        # Weighted selection: accumulate and threshold
        weights = [0.12, 0.15, 0.18, 0.18, 0.15, 0.10, 0.07, 0.05]
        r = rng.uniform(0, 1)
        cumulative = 0
        idx = 0
        for wi, w in enumerate(weights):
            cumulative += w
            if r < cumulative:
                idx = wi
                break
        freq = freq_list[idx]

        # Tiny microtonal drift — wind doesn't hit perfectly
        freq *= 2 ** (rng.uniform(-8, 8) / 1200.0)

        # Soft, varied dynamics — wind gusts
        amp = rng.uniform(0.18, 0.42)

        # Let each bell ring for 10 seconds — the exponential decay
        # in the partials will kill it naturally well before that.
        # This prevents the hard cutoff from the mask in bell_strike.
        note_dur = 10.0

        audio += bell_strike(t, current_time, freq, note_dur, amp, WIND_TUBE, rng)
        strike_count += 1

        # Next strike: irregular wind timing
        # Sometimes quick double-taps, sometimes long silences
        r = rng.uniform(0, 1)
        if r < 0.15:
            gap = rng.uniform(0.3, 0.7)   # quick follow-up gust
        elif r < 0.5:
            gap = rng.uniform(1.2, 2.5)   # normal breeze
        else:
            gap = rng.uniform(2.8, 4.5)   # lull

        current_time += gap

    print(f"  {strike_count} bell strikes over {content_time:.0f} seconds")

    # Cathedral reverb — lots of space
    audio = simple_reverb(audio, preset="cathedral", sample_rate=SAMPLE_RATE)

    # Gentle fade in and long fade out
    fade_in = int(2.0 * SAMPLE_RATE)
    audio[:fade_in] *= np.linspace(0, 1, fade_in)
    fade_out = int(6.0 * SAMPLE_RATE)
    audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    # Normalize gently — keep it soft
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.82

    return audio, duration


def main():
    print("Wind Bells — tubular bells stirred by breeze")
    print()
    audio, dur = generate()

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "wind_bells.wav")
    mp3_path = os.path.join(out_dir, "audio", "wind_bells.mp3")

    audio_16 = np.int16(audio * 32767)
    wavfile.write(wav_path, SAMPLE_RATE, audio_16)
    print(f"  Written: {wav_path}")

    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-codec:a', 'libmp3lame', '-b:a', '192k', mp3_path],
        capture_output=True
    )
    print(f"  Written: {mp3_path}")


if __name__ == "__main__":
    main()
