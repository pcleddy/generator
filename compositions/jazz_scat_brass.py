"""
jazz_scat_brass.py — Jazz scat demo for FM brass instruments.

A muted trumpet plays a bebop-style scat line over walking bass (fm_tuba),
with fm_horn comping chords and fm_flugelhorn taking a lyrical counter-melody.
The trombone drops in for punctuation hits.

Blues in Bb: Bb7 | Eb7 | Bb7 | Bb7 | Eb7 | Eb7 | Bb7 | G7 | Cm7 | F7 | Bb7 | F7

Swing feel: every other 8th note slightly delayed (swing ratio ~0.62/0.38).
Tempo: 160 BPM (bright bebop tempo).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis_engine import Renderer, SeedManager, SynthEvent, InstrumentRegistry
from synthesis_engine.config import SAMPLE_RATE

# ── Constants ──
TEMPO = 160  # BPM
BEAT = 60.0 / TEMPO  # seconds per beat
EIGHTH = BEAT / 2
SWING_LONG = BEAT * 0.62    # swing 8th (long)
SWING_SHORT = BEAT * 0.38   # swing 8th (short)
TRIPLET = BEAT / 3

# Pitch classes: C=0 C#=1 D=2 Eb=3 E=4 F=5 F#=6 G=7 Ab=8 A=9 Bb=10 B=11

# ── Blues scale in Bb ──
# Bb C Db D F Gb (blues) + bebop passing tones
BLUES_SCALE = [10, 0, 1, 2, 5, 6]  # Bb blues
BEBOP_SCALE = [10, 0, 1, 2, 3, 5, 6, 7, 9]  # Bb bebop with chromatic passing

# ── Chord tones for each chord ──
CHORDS = {
    "Bb7":  [10, 2, 5, 8],   # Bb D F Ab
    "Eb7":  [3, 5, 8, 1],    # Eb G Bb Db
    "G7":   [7, 11, 2, 5],   # G B D F
    "Cm7":  [0, 3, 7, 10],   # C Eb G Bb
    "F7":   [5, 9, 0, 3],    # F A C Eb
}

# 12-bar blues progression (each entry = 1 bar = 4 beats)
PROGRESSION = [
    "Bb7", "Eb7", "Bb7", "Bb7",
    "Eb7", "Eb7", "Bb7", "G7",
    "Cm7", "F7",  "Bb7", "F7",
]


def generate():
    rng = SeedManager(42)
    events = []
    bar_dur = 4 * BEAT  # 4 beats per bar

    # We'll do 2 choruses (24 bars)
    total_bars = 24

    # ── MUTED TRUMPET — scat melody line ──
    t = 0.0
    for bar_idx in range(total_bars):
        chord_name = PROGRESSION[bar_idx % 12]
        chord_tones = CHORDS[chord_name]
        bar_start = bar_idx * bar_dur

        # Generate a scat phrase for this bar
        # Vary density: some bars busy (8ths), some sparse (quarter + rest)
        density = rng.uniform(0.3, 1.0)

        if density > 0.7:
            # Busy bar: bebop 8th note line
            t = bar_start
            for beat in range(4):
                for sub in range(2):  # two 8ths per beat
                    if rng.random() < 0.85:  # occasional rest
                        # Pick note: mostly chord tones, some passing
                        if rng.random() < 0.6:
                            pc = rng.choice(chord_tones)
                        else:
                            pc = rng.choice(BEBOP_SCALE)
                        octave = rng.choice([4, 5]) if rng.random() < 0.7 else 4
                        dur = SWING_LONG if sub == 0 else SWING_SHORT
                        # Scat articulation: short notes with accents
                        note_dur = dur * rng.uniform(0.5, 0.9)
                        amp = rng.uniform(0.35, 0.6)
                        # Ghost notes (very soft passing tones)
                        if rng.random() < 0.2:
                            amp *= 0.4
                        events.append(SynthEvent(
                            time=t, pitch_class=pc, octave=octave,
                            duration=note_dur, amplitude=amp,
                            instrument="fm_muted_trumpet",
                            category="scat", section="melody",
                        ))
                    t += SWING_LONG if sub == 0 else SWING_SHORT

        elif density > 0.4:
            # Medium bar: quarter notes with some 8ths
            t = bar_start
            for beat in range(4):
                if rng.random() < 0.7:
                    pc = rng.choice(chord_tones)
                    octave = 4 if beat < 2 else rng.choice([4, 5])
                    note_dur = BEAT * rng.uniform(0.6, 0.95)
                    amp = rng.uniform(0.4, 0.55)
                    events.append(SynthEvent(
                        time=t, pitch_class=pc, octave=octave,
                        duration=note_dur, amplitude=amp,
                        instrument="fm_muted_trumpet",
                        category="scat", section="melody",
                    ))
                t += BEAT

        else:
            # Sparse bar: one or two long notes (held tones, vibrato)
            t = bar_start + BEAT * rng.uniform(0, 1)
            pc = rng.choice(chord_tones)
            note_dur = BEAT * rng.uniform(2, 3.5)
            events.append(SynthEvent(
                time=t, pitch_class=pc, octave=4,
                duration=note_dur, amplitude=0.5,
                instrument="fm_muted_trumpet",
                category="scat", section="melody",
            ))

    # ── FM TUBA — walking bass line ──
    for bar_idx in range(total_bars):
        chord_name = PROGRESSION[bar_idx % 12]
        chord_tones = CHORDS[chord_name]
        bar_start = bar_idx * bar_dur

        for beat in range(4):
            t = bar_start + beat * BEAT
            if beat == 0:
                # Root on beat 1
                pc = chord_tones[0]
            elif beat == 2:
                # Fifth on beat 3
                pc = chord_tones[2]  # 5th
            else:
                # Walking: chromatic approach or scale step
                if rng.random() < 0.5:
                    pc = rng.choice(chord_tones)
                else:
                    # Chromatic approach to next beat's target
                    pc = rng.choice(BLUES_SCALE)

            note_dur = BEAT * rng.uniform(0.7, 0.95)
            amp = 0.45 if beat in (0, 2) else 0.35  # accent 1 and 3
            events.append(SynthEvent(
                time=t, pitch_class=pc, octave=2,
                duration=note_dur, amplitude=amp,
                instrument="fm_tuba",
                category="bass", section="walking_bass",
            ))

    # ── FM HORN — chord comps (on 2 and 4, like a rhythm section) ──
    for bar_idx in range(total_bars):
        chord_name = PROGRESSION[bar_idx % 12]
        chord_tones = CHORDS[chord_name]
        bar_start = bar_idx * bar_dur

        for beat in [1, 3]:  # beats 2 and 4 (0-indexed)
            t = bar_start + beat * BEAT
            # Play 2-3 chord tones simultaneously (voicing)
            n_notes = rng.choice([2, 3])
            voicing = rng.choice(chord_tones)  # can't do multi-select easily
            # Just play the chord root and third as a dyad
            for i, pc in enumerate(chord_tones[:n_notes]):
                octave = 3 if i == 0 else 4
                note_dur = BEAT * rng.uniform(0.3, 0.6)  # short stabs
                amp = rng.uniform(0.2, 0.35)  # behind the melody
                events.append(SynthEvent(
                    time=t + rng.uniform(-0.01, 0.01),  # slight spread
                    pitch_class=pc, octave=octave,
                    duration=note_dur, amplitude=amp,
                    instrument="fm_horn",
                    category="comp", section="chords",
                ))

    # ── FM FLUGELHORN — counter-melody (2nd chorus only, sparser) ──
    for bar_idx in range(12, 24):  # 2nd chorus
        chord_name = PROGRESSION[bar_idx % 12]
        chord_tones = CHORDS[chord_name]
        bar_start = bar_idx * bar_dur

        if rng.random() < 0.5:  # only play half the bars
            # One or two lyrical notes
            t = bar_start + BEAT * rng.uniform(0.5, 2.0)
            pc = rng.choice(chord_tones)
            note_dur = BEAT * rng.uniform(1.5, 3.0)
            events.append(SynthEvent(
                time=t, pitch_class=pc, octave=4,
                duration=note_dur, amplitude=0.3,
                instrument="fm_flugelhorn",
                category="counter", section="flugelhorn",
            ))

    # ── FM TROMBONE — punctuation hits ──
    # Big hits at turnaround bars (bars 11-12, 23-24) and a few riffs
    for chorus in [0, 1]:
        base = chorus * 12
        for bar_offset in [10, 11]:  # turnaround
            bar_idx = base + bar_offset
            chord_name = PROGRESSION[bar_idx % 12]
            chord_tones = CHORDS[chord_name]
            bar_start = bar_idx * bar_dur

            # Trombone hits on beat 1 and the "and" of 2
            for hit_time, dur_mult in [(0, 1.5), (1.5 * BEAT, 0.8)]:
                t = bar_start + hit_time
                pc = chord_tones[rng.randint(0, len(chord_tones) - 1)]
                events.append(SynthEvent(
                    time=t, pitch_class=pc, octave=3,
                    duration=BEAT * dur_mult,
                    amplitude=rng.uniform(0.5, 0.65),
                    instrument="fm_trombone",
                    category="hits", section="trombone",
                ))

        # A short riff in bar 7 (the G7 bar)
        bar_start = (base + 7) * bar_dur
        riff_notes = [(7, 3), (11, 3), (2, 4), (5, 3)]  # G B D F descending
        t = bar_start
        for pc, oct in riff_notes:
            events.append(SynthEvent(
                time=t, pitch_class=pc, octave=oct,
                duration=BEAT * 0.4, amplitude=0.45,
                instrument="fm_trombone",
                category="riff", section="trombone",
            ))
            t += SWING_LONG

    # ── FM TRUMPET (open) — final 4 bars: blazing solo ──
    for bar_idx in range(20, 24):
        chord_name = PROGRESSION[bar_idx % 12]
        chord_tones = CHORDS[chord_name]
        bar_start = bar_idx * bar_dur

        t = bar_start
        for beat in range(4):
            for sub in range(2):
                if rng.random() < 0.9:
                    pc = rng.choice(BEBOP_SCALE) if rng.random() > 0.4 else rng.choice(chord_tones)
                    octave = rng.choice([4, 5, 5])  # higher register
                    dur = SWING_LONG if sub == 0 else SWING_SHORT
                    note_dur = dur * rng.uniform(0.4, 0.85)
                    amp = rng.uniform(0.45, 0.7)
                    events.append(SynthEvent(
                        time=t, pitch_class=pc, octave=octave,
                        duration=note_dur, amplitude=amp,
                        instrument="fm_trumpet",
                        category="solo", section="trumpet_solo",
                    ))
                t += SWING_LONG if sub == 0 else SWING_SHORT

    total_dur = total_bars * bar_dur + 3.0  # 3s reverb tail
    return events, total_dur


def main():
    events, duration = generate()

    print(f"Jazz Scat Brass: {len(events)} events, {duration:.1f}s")
    print(f"  Tempo: {TEMPO} BPM, {len(PROGRESSION)}-bar blues in Bb × 2 choruses")

    # Count by instrument
    counts = {}
    for e in events:
        counts[e.instrument] = counts.get(e.instrument, 0) + 1
    for inst, n in sorted(counts.items()):
        print(f"  {inst}: {n} events")

    registry = InstrumentRegistry()
    renderer = Renderer(registry=registry)
    rng = SeedManager(42)

    audio = renderer.render(events, duration, rng=rng, reverb_preset="intimate")

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(out_dir, "audio", "jazz_scat_brass.wav")
    mp3_path = os.path.join(out_dir, "audio", "jazz_scat_brass.mp3")
    json_path = os.path.join(out_dir, "scores", "jazz_scat_brass.json")

    renderer.save_wav(audio, wav_path)
    renderer.save_mp3(wav_path, mp3_path)
    print(f"  Written: {mp3_path}")

    import json
    with open(json_path, "w") as f:
        json.dump({
            "title": "Jazz Scat — FM Brass",
            "duration": duration,
            "events": [e.to_dict() for e in events],
            "audio_file": "jazz_scat_brass.mp3",
            "seed": 42,
        }, f, indent=2)
    print(f"  Written: {json_path}")


if __name__ == "__main__":
    main()
