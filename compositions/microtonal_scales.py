"""
microtonal_scales.py — Nylon guitar running up and down a Phrygian scale,
modulating upward by microtones each pass.

Starts in E Phrygian at standard tuning. Each ascending/descending pass
drifts the whole scale up by ~15-20 cents. By the end of 30 seconds
the pitch has risen about a whole tone — E Phrygian has become F Phrygian
but arrived there through the cracks between the keys.

Direct synthesis — bypasses the Renderer to control exact frequencies.
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

DURATION = 32.0  # total piece length
PROFILE = PLUCKED_PROFILES["guitar_nylon"]

# E Phrygian intervals in semitones from root: 0, 1, 3, 5, 7, 8, 10, 12
PHRYGIAN_INTERVALS = [0, 1, 3, 5, 7, 8, 10, 12]

# Base frequency: E3 = 164.81 Hz
E3_FREQ = 164.81


def cents_to_ratio(cents):
    return 2 ** (cents / 1200.0)


def generate():
    rng = SeedManager(42)
    t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE), endpoint=False)
    audio = np.zeros(len(t))

    current_time = 0.2
    microtone_offset = 0.0  # cumulative cents drift
    pass_num = 0

    while current_time < DURATION - 2.0:
        # Each pass: ascending then descending
        ascending = PHRYGIAN_INTERVALS.copy()
        descending = list(reversed(PHRYGIAN_INTERVALS[:-1]))  # don't repeat top note

        # Vary speed: starts moderate, gets faster, then slows at end
        progress = current_time / DURATION
        if progress < 0.3:
            base_note_dur = 0.18 - progress * 0.2  # 0.18 → 0.12
        elif progress < 0.7:
            base_note_dur = 0.10 - (progress - 0.3) * 0.08  # 0.10 → 0.07
        else:
            base_note_dur = 0.07 + (progress - 0.7) * 0.3  # 0.07 → 0.16 (slow down)

        # Vary octave: mostly E3, occasionally jump to E4
        if pass_num % 5 == 3:
            base_octave_mult = 2.0  # octave up
        elif pass_num % 7 == 4:
            base_octave_mult = 0.5  # octave down (E2 — deep)
        else:
            base_octave_mult = 1.0

        # Vary amplitude: slight crescendo-decrescendo per pass
        scale = ascending + descending

        for i, interval in enumerate(scale):
            if current_time >= DURATION - 1.5:
                break

            # Frequency: base note + scale interval + microtone drift
            total_semitones = interval
            freq = E3_FREQ * (2 ** (total_semitones / 12.0)) * base_octave_mult
            freq *= cents_to_ratio(microtone_offset)

            # Add tiny per-note humanization (±3 cents)
            freq *= cents_to_ratio(rng.uniform(-3, 3))

            # Duration with slight swing feel
            note_dur = base_note_dur * rng.uniform(0.85, 1.15)

            # Amplitude: arc shape per pass, louder at top
            arc = 1.0 - abs(i - len(scale)/2) / (len(scale)/2)
            amp = 0.35 + 0.25 * arc
            amp *= rng.uniform(0.9, 1.1)

            # Synthesize
            audio += karplus_strong(
                t, current_time, freq, note_dur, min(amp, 0.65),
                PROFILE, rng
            )

            current_time += note_dur

        # After each pass, modulate up by microtones
        # Amount varies: sometimes 12 cents, sometimes 25, occasionally 40
        drift = rng.uniform(12, 28)
        # Accelerate the drift slightly as we go
        drift *= (1.0 + pass_num * 0.05)
        microtone_offset += drift

        # Brief pause between passes (varies)
        current_time += rng.uniform(0.08, 0.25)
        pass_num += 1

        # Print progress
        if pass_num % 4 == 0:
            print(f"  Pass {pass_num}: t={current_time:.1f}s, "
                  f"drift={microtone_offset:.0f} cents "
                  f"({microtone_offset/100:.1f} semitones)")

    # Final chord: let the last note ring out with no more notes
    # Play the root at whatever pitch we've drifted to
    final_freq = E3_FREQ * cents_to_ratio(microtone_offset)
    audio += karplus_strong(t, current_time, final_freq, 3.0, 0.55, PROFILE, rng)
    # Add the fifth above
    audio += karplus_strong(t, current_time + 0.03, final_freq * 1.5, 3.0, 0.4, PROFILE, rng)

    print(f"\n  Total: {pass_num} passes, final drift: {microtone_offset:.0f} cents "
          f"({microtone_offset/100:.1f} semitones)")

    # Apply reverb
    audio = simple_reverb(audio, preset="intimate", sample_rate=SAMPLE_RATE)

    # Fade end
    fade_len = int(1.5 * SAMPLE_RATE)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.92

    return audio


def main():
    print("Microtonal Scales — Nylon Guitar")
    print(f"  E Phrygian, drifting upward by microtones, {DURATION}s")
    print()

    audio = generate()

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "microtonal_scales.wav")
    mp3_path = os.path.join(out_dir, "audio", "microtonal_scales.mp3")

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
