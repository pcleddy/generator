"""
voice_analysis.py — Spectral analysis of synthesized voices

Generates spectrograms and frequency snapshots to visualize
what our formant synthesis is actually doing vs what real
voices look like. Diagnostic tool for tuning.

Outputs: voice_analysis.png
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(filename="voice_demo_03.wav"):
    sr, data = wavfile.read(filename)
    audio = data.astype(np.float64) / 32767.0

    # Time markers for each voice in the demo
    segments = {
        "Bass G2":      (1.5,   8.5),
        "Tenor A3":     (11.0,  18.0),
        "Alto C4":      (20.5,  27.5),
        "Soprano E5":   (30.0,  37.0),
        'Vowel "ah"':   (41.5,  48.5),
        'Vowel "ee"':   (51.0,  58.0),
        'Vowel "oh"':   (60.5,  67.5),
        'Vowel "oo"':   (70.0,  77.0),
        'Vowel "eh"':   (79.5,  86.5),
        'Vowel "mm"':   (89.0,  96.0),
    }

    fig, axes = plt.subplots(5, 2, figsize=(18, 22))
    fig.suptitle("Voice Synthesis — Spectral Analysis", fontsize=16, fontweight='bold')
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    for idx, (label, (t_start, t_end)) in enumerate(segments.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        s_start = int(t_start * sr)
        s_end = min(int(t_end * sr), len(audio))
        segment = audio[s_start:s_end]

        # Compute spectrogram
        f, t_spec, Sxx = spectrogram(segment, fs=sr, nperseg=2048,
                                      noverlap=1536, window='hann')

        # Limit to 0-5kHz (voice range)
        freq_mask = f <= 5000
        Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)

        ax.pcolormesh(t_spec, f[freq_mask], Sxx_db,
                       shading='gouraud', cmap='inferno', vmin=-80, vmax=-20)
        ax.set_ylabel('Hz')
        ax.set_xlabel('Time (s)')
        ax.set_title(label, fontsize=12, fontweight='bold')

        # Mark expected formant frequencies for vowels
        formant_markers = {
            'Vowel "ah"': [800, 1150, 2800],
            'Vowel "ee"': [270, 2300, 3000],
            'Vowel "oh"': [500, 700, 2800],
            'Vowel "oo"': [300, 600, 2300],
            'Vowel "eh"': [530, 1850, 2500],
            'Vowel "mm"': [250, 1700, 2500],
        }
        if label in formant_markers:
            for ff in formant_markers[label]:
                ax.axhline(y=ff, color='cyan', alpha=0.5, linewidth=0.8, linestyle='--')

    out = "voice_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

    # Also do a single-frame frequency snapshot (FFT) for alto "ah" vs "ee"
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle('Frequency Spectrum Snapshot — Alto C4', fontsize=14, fontweight='bold')

    for ax, label, (t_start, t_end) in [
        (ax1, '"ah" (F1=800, F2=1150)', (41.5, 48.5)),
        (ax2, '"ee" (F1=270, F2=2300)', (51.0, 58.0)),
    ]:
        s_start = int((t_start + 2.0) * sr)  # 2s in — past attack
        s_end = s_start + 4096
        if s_end > len(audio):
            s_end = len(audio)
        chunk = audio[s_start:s_end]

        # Windowed FFT
        window = np.hanning(len(chunk))
        fft = np.abs(np.fft.rfft(chunk * window))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        # Limit to 0-5kHz
        mask = freqs <= 5000
        fft_db = 20 * np.log10(fft[mask] + 1e-12)

        ax.plot(freqs[mask], fft_db, color='#ff6600', linewidth=0.6)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dB')
        ax.set_title(label)
        ax.set_ylim(-80, 0)
        ax.grid(True, alpha=0.3)

    out2 = "voice_spectrum.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out2}")

if __name__ == "__main__":
    analyze()
