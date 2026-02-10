"""
Analyze Paul's voice recording to extract:
- Fundamental frequency (F0)
- Formant frequencies (F1-F4)
- Spectral envelope shape
- Harmonic structure
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, find_peaks

sr, data = wavfile.read("real_voice_sample.wav")
audio = data.astype(np.float64) / 32767.0

# Grab a stable voiced section (middle of recording)
mid = len(audio) // 2
chunk_len = int(0.1 * sr)  # 100ms window
chunk = audio[mid - chunk_len:mid + chunk_len]

# --- F0 estimation via autocorrelation ---
# Window and autocorrelate
windowed = chunk * np.hanning(len(chunk))
corr = np.correlate(windowed, windowed, mode='full')
corr = corr[len(corr)//2:]  # positive lags only
corr /= corr[0]  # normalize

# Find first peak after minimum pitch period (50 Hz = 882 samples at 44100)
min_lag = int(sr / 400)  # 400 Hz max
max_lag = int(sr / 60)   # 60 Hz min

search = corr[min_lag:max_lag]
peaks, props = find_peaks(search, height=0.3)

if len(peaks) > 0:
    best_lag = peaks[0] + min_lag
    f0 = sr / best_lag
else:
    f0 = 0

print(f"=== Paul's Voice Analysis ===\n")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Estimated F0: {f0:.1f} Hz")

if f0 > 0:
    # What note is this?
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    semitones_from_a4 = 12 * np.log2(f0 / 440.0)
    note_idx = int(round(semitones_from_a4)) % 12
    octave = int(4 + (round(semitones_from_a4) + 9) // 12) - 1
    print(f"Nearest note: {note_names[(note_idx + 9) % 12]}{octave}")

# --- Spectral envelope via LPC (formant estimation) ---
# LPC gives us the all-pole model of the vocal tract
from numpy.linalg import solve

def lpc_coeffs(signal, order):
    """Compute LPC coefficients via autocorrelation method."""
    n = len(signal)
    r = np.correlate(signal, signal, mode='full')
    r = r[n-1:]  # positive lags
    # Levinson-Durbin
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    rhs = r[1:order + 1]
    a = solve(R, rhs)
    return a

# Use a shorter chunk for LPC (25ms is standard for speech)
lpc_chunk_len = int(0.025 * sr)
lpc_chunk = audio[mid:mid + lpc_chunk_len]
lpc_chunk = lpc_chunk * np.hanning(len(lpc_chunk))

# Pre-emphasis to flatten spectral tilt
pre_emph = np.append(lpc_chunk[0], lpc_chunk[1:] - 0.97 * lpc_chunk[:-1])

# LPC order: rule of thumb = sr/1000 + 2
lpc_order = sr // 1000 + 2
a = lpc_coeffs(pre_emph, lpc_order)

# Find formants from LPC poles
roots = np.roots(np.concatenate(([1], -a)))
# Only keep roots inside unit circle with positive angle
roots = roots[np.imag(roots) >= 0]
roots = roots[np.abs(roots) < 1.0]

# Convert to frequencies
angles = np.angle(roots)
freqs = angles * sr / (2 * np.pi)
bws = -sr / (2 * np.pi) * np.log(np.abs(roots))

# Filter: formants are between 90 Hz and 5000 Hz with reasonable bandwidth
formants = []
for f, bw in zip(freqs, bws):
    if 90 < f < 5000 and bw < 500:
        formants.append((f, bw))

formants.sort(key=lambda x: x[0])

print(f"\nEstimated formants (LPC order {lpc_order}):")
for i, (f, bw) in enumerate(formants[:5]):
    label = f"F{i+1}"
    print(f"  {label}: {f:.0f} Hz  (bandwidth: {bw:.0f} Hz)")

# --- Spectral envelope for reference ---
fft_chunk = audio[mid:mid + 4096]
window = np.hanning(len(fft_chunk))
fft = np.abs(np.fft.rfft(fft_chunk * window))
freqs_fft = np.fft.rfftfreq(len(fft_chunk), 1.0 / sr)
fft_db = 20 * np.log10(fft + 1e-12)

# Find spectral peaks
mask_5k = freqs_fft <= 5000
peaks_spec, _ = find_peaks(fft_db[mask_5k], height=-30, distance=20, prominence=5)
peak_freqs = freqs_fft[mask_5k][peaks_spec]
peak_amps = fft_db[mask_5k][peaks_spec]

print(f"\nSpectral peaks (prominent):")
for f, a in sorted(zip(peak_freqs, peak_amps), key=lambda x: -x[1])[:10]:
    print(f"  {f:.0f} Hz  ({a:.1f} dB)")

# Harmonic spacing
print(f"\nHarmonic spacing (if F0={f0:.0f} Hz):")
for h in range(1, 16):
    expected = f0 * h
    if expected > 5000:
        break
    # Find nearest spectral peak
    dists = np.abs(peak_freqs - expected)
    if len(dists) > 0 and np.min(dists) < 30:
        nearest = peak_freqs[np.argmin(dists)]
        amp = peak_amps[np.argmin(dists)]
        print(f"  H{h:2d}: expected {expected:.0f} Hz, found {nearest:.0f} Hz ({amp:.1f} dB)")
