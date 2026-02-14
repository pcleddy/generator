"""
microtonal_slow.py — Slow microtonal scale: one note at a time, 2s decay each.

Nylon guitar playing E Phrygian, one note every 2 seconds so you can hear
each note's full Karplus-Strong decay. Each pass through the scale drifts
the pitch up by ~20 cents. Simple and clear.

Direct synthesis — bypasses the Renderer for exact frequency control.
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

PROFILE = PLUCKED_PROFILES["guitar_nylon"]

# E Phrygian intervals in semitones: E F G A B C D E
PHRYGIAN_INTERVALS = [0, 1, 3, 5, 7, 8, 10, 12]

# Base frequency: E3 = 164.81 Hz
E3_FREQ = 164.81

NOTE_SPACING = 2.0  # seconds between notes


def cents_to_ratio(cents):
    return 2 ** (cents / 1200.0)


def generate():
    rng = SeedManager(42)

    # Ascending then descending = 15 notes per pass
    # 3 passes × 15 notes × 2s = 90s, but let's do 2 passes to keep it ~60s
    ascending = PHRYGIAN_INTERVALS.copy()
    descending = list(reversed(PHRYGIAN_INTERVALS[:-1]))  # skip repeated top note
    one_pass = ascending + descending  # 15 notes

    n_passes = 2
    total_notes = n_passes * len(one_pass)
    duration = total_notes * NOTE_SPACING + 4.0  # extra for final ring-out

    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    audio = np.zeros(len(t))

    current_time = 0.5  # small lead-in
    microtone_offset = 0.0

    for pass_num in range(n_passes):
        for i, interval in enumerate(one_pass):
            # Frequency: base + scale interval + microtone drift
            freq = E3_FREQ * (2 ** (interval / 12.0))
            freq *= cents_to_ratio(microtone_offset)

            # Tiny humanization (±2 cents)
            freq *= cents_to_ratio(rng.uniform(-2, 2))

            # Amplitude: consistent, slight variation
            amp = 0.55 * rng.uniform(0.95, 1.05)

            # Synthesize with long duration so it can ring
            audio += karplus_strong(
                t, current_time, freq, NOTE_SPACING * 0.9, min(amp, 0.65),
                PROFILE, rng
            )

            current_time += NOTE_SPACING

        # After each pass, drift up by ~20 cents
        drift = rng.uniform(18, 25)
        microtone_offset += drift
        print(f"  Pass {pass_num + 1}: drift now {microtone_offset:.0f} cents "
              f"({microtone_offset / 100:.2f} semitones)")

        # Brief extra pause between passes
        current_time += 1.0

    # Final note: root at current drift
    final_freq = E3_FREQ * cents_to_ratio(microtone_offset)
    audio += karplus_strong(t, current_time, final_freq, 3.0, 0.6, PROFILE, rng)

    print(f"\n  Total: {n_passes} passes, {total_notes} notes, "
          f"final drift: {microtone_offset:.0f} cents")

    # Light reverb
    audio = simple_reverb(audio, preset="intimate", sample_rate=SAMPLE_RATE)

    # Fade end
    fade_len = int(2.0 * SAMPLE_RATE)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.92

    return audio, duration


def main():
    print("Microtonal Slow — Nylon Guitar")
    print(f"  E Phrygian, one note every {NOTE_SPACING}s, drifting upward")
    print()

    audio, duration = generate()

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "microtonal_slow.wav")
    mp3_path = os.path.join(out_dir, "audio", "microtonal_slow.mp3")

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
