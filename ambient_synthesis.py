import numpy as np
from scipy.io import wavfile
from scipy import signal
import random

# Parameters
SAMPLE_RATE = 44100
DURATION = 120  # seconds
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

# Initialize output
audio = np.zeros_like(t)

# --- Pitch material ---
# Define primary 12-tone row
twelve_tone = np.array([0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10])

# Inversion (mirror around C)
twelve_tone_inv = (12 - twelve_tone) % 12

# Base frequency
BASE_FREQ = 130.81

def freq_from_semitones(semitones, octave_shift=0):
    """Convert semitones to frequency"""
    return BASE_FREQ * (2 ** ((semitones + octave_shift * 12) / 12))

# --- Advanced oscillator with harmonics and modulation ---
def harmonic_voice(start_time, pitch_semitones, octave, duration, fade_in, fade_out, 
                   harmonic_mix=None, amplitude=0.08, mod_freq=None, mod_depth=0):
    """Generate oscillator with subtle harmonics and optional frequency modulation"""
    voice = np.zeros_like(t)
    freq = freq_from_semitones(pitch_semitones, octave)
    
    if harmonic_mix is None:
        harmonic_mix = [1.0]
    
    mask = (t >= start_time) & (t < start_time + duration)
    t_local = t[mask] - start_time
    
    harmonic_out = np.zeros_like(t_local)
    for harm_num, harm_amp in enumerate(harmonic_mix, 1):
        freq_mod = 0
        if mod_freq is not None:
            freq_mod = mod_depth * np.sin(2 * np.pi * mod_freq * t_local)
        
        harmonic_out += harm_amp * np.sin(2 * np.pi * (freq * harm_num + freq_mod) * t_local)
    
    voice[mask] = harmonic_out / len(harmonic_mix)
    
    fade_in_mask = (t >= start_time) & (t < start_time + fade_in)
    fade_out_mask = (t >= start_time + duration - fade_out) & (t < start_time + duration)
    
    voice[fade_in_mask] *= (t[fade_in_mask] - start_time) / fade_in
    voice[fade_out_mask] *= (start_time + duration - t[fade_out_mask]) / fade_out
    
    return voice * amplitude

# Layer 1: Primary 12-tone row
for i, pitch in enumerate(twelve_tone):
    start = i * 5.5
    if start + 6 < DURATION:
        audio += harmonic_voice(start, pitch, octave=0, duration=6, 
                              fade_in=2, fade_out=2, 
                              harmonic_mix=[0.85, 0.15], amplitude=0.07)

# Layer 2: Inverted row
for i, pitch in enumerate(twelve_tone_inv):
    start = 3 + i * 5.5
    if start + 6 < DURATION:
        audio += harmonic_voice(start, pitch, octave=0, duration=6, 
                              fade_in=2.5, fade_out=2.5, 
                              harmonic_mix=[0.8, 0.2], amplitude=0.06)

# Layer 3: Microtonal chaos
random.seed(42)
for i in range(24):
    start_time = random.uniform(0, DURATION - 10)
    if random.random() < 0.4:
        base_pitch = random.choice(twelve_tone)
        microtone_offset = random.uniform(-30, 30) / 100
    else:
        base_pitch = random.uniform(0, 12)
        microtone_offset = random.uniform(-50, 50) / 100
    
    final_pitch = base_pitch + microtone_offset
    duration = random.uniform(8, 20)
    octave_shift = random.choice([-2, -1, 0])
    mod_freq = random.uniform(0.05, 0.2)
    mod_depth = random.uniform(0.5, 3)
    
    audio += harmonic_voice(start_time, final_pitch, octave=octave_shift, 
                           duration=duration, fade_in=3, fade_out=4,
                           harmonic_mix=[0.9, 0.1], amplitude=0.04,
                           mod_freq=mod_freq, mod_depth=mod_depth)

# Layer 4: Sub-bass
low_pitches = [-12, -10, -7, -5, -2]
for pitch in low_pitches:
    start = np.random.uniform(0, DURATION - 15)
    duration = np.random.uniform(18, DURATION - start)
    mod_freq = random.uniform(0.02, 0.08)
    
    audio += harmonic_voice(start, pitch, octave=0, duration=duration, 
                           fade_in=4, fade_out=5, 
                           harmonic_mix=[1.0, 0.05],
                           amplitude=0.035, mod_freq=mod_freq, mod_depth=2)

# Layer 5: Clusters
for cluster_time in np.arange(15, DURATION - 10, 18):
    base = random.choice(twelve_tone)
    cluster_size = random.randint(2, 4)
    
    for voice_idx in range(cluster_size):
        detune = random.uniform(-15, 15) / 100
        delay = voice_idx * 1.2
        
        audio += harmonic_voice(cluster_time + delay, base + detune, octave=-1, 
                               duration=8, fade_in=2.5, fade_out=3,
                               harmonic_mix=[0.9, 0.1], amplitude=0.045,
                               mod_freq=0.06, mod_depth=1.5)

# Finalize
audio = audio / (np.max(np.abs(audio)) * 1.05)
audio = np.tanh(audio * 1.2) / 1.2

wavfile.write('ambient.wav', SAMPLE_RATE, (audio * 32767).astype(np.int16))
print("Generated: ambient.wav")
