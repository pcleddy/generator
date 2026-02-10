"""
compare_voice.py — Compare real voice recording vs our synthesis

Side-by-side spectrograms and FFT snapshots showing the gap
between a real human voice and our formant synthesis.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_comparison():
    # Load real voice
    sr_real, data_real = wavfile.read("real_voice_sample.wav")
    real = data_real.astype(np.float64) / 32767.0

    # Load our synthesis (alto "ah" from voice demo)
    sr_synth, data_synth = wavfile.read("voice_demo_03.wav")
    synth = data_synth.astype(np.float64) / 32767.0

    # ---- Figure 1: Spectrograms side by side ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Real Voice vs Synthesis — Spectral Comparison",
                 fontsize=16, fontweight='bold')

    # Real voice spectrogram
    f, t_s, Sxx = spectrogram(real, fs=sr_real, nperseg=2048,
                               noverlap=1536, window='hann')
    freq_mask = f <= 5000
    Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)

    axes[0, 0].pcolormesh(t_s, f[freq_mask], Sxx_db,
                           shading='gouraud', cmap='inferno', vmin=-80, vmax=-10)
    axes[0, 0].set_title("REAL VOICE (your recording)", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("Hz")
    axes[0, 0].set_xlabel("Time (s)")

    # Synth voice spectrogram — alto "ah" (starts at 41.5s)
    synth_start = int(41.5 * sr_synth)
    synth_end = int(48.5 * sr_synth)
    synth_segment = synth[synth_start:synth_end]

    f2, t_s2, Sxx2 = spectrogram(synth_segment, fs=sr_synth, nperseg=2048,
                                   noverlap=1536, window='hann')
    freq_mask2 = f2 <= 5000
    Sxx2_db = 10 * np.log10(Sxx2[freq_mask2] + 1e-12)

    axes[0, 1].pcolormesh(t_s2, f2[freq_mask2], Sxx2_db,
                           shading='gouraud', cmap='inferno', vmin=-80, vmax=-10)
    axes[0, 1].set_title('SYNTHESIS — Alto "ah" (C4)', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel("Hz")
    axes[0, 1].set_xlabel("Time (s)")

    # ---- FFT Snapshots ----
    # Real voice — grab a stable chunk from the middle
    real_mid = len(real) // 2
    real_chunk = real[real_mid - 4096:real_mid + 4096]
    if len(real_chunk) < 1024:
        real_chunk = real[:8192]

    window_r = np.hanning(len(real_chunk))
    fft_real = np.abs(np.fft.rfft(real_chunk * window_r))
    freqs_real = np.fft.rfftfreq(len(real_chunk), 1.0 / sr_real)
    mask_r = freqs_real <= 5000
    fft_real_db = 20 * np.log10(fft_real[mask_r] + 1e-12)

    axes[1, 0].plot(freqs_real[mask_r], fft_real_db, color='#ff6600', linewidth=0.6)
    axes[1, 0].set_title("REAL VOICE — Frequency Spectrum (mid-recording)",
                          fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].set_ylim(-80, 10)
    axes[1, 0].grid(True, alpha=0.3)

    # Synth voice — grab stable chunk from "ah" section
    synth_mid_start = int(43.5 * sr_synth)  # 2s into "ah"
    synth_chunk = synth[synth_mid_start:synth_mid_start + 8192]
    if len(synth_chunk) < 1024:
        synth_chunk = synth_segment[:8192]

    window_s = np.hanning(len(synth_chunk))
    fft_synth = np.abs(np.fft.rfft(synth_chunk * window_s))
    freqs_synth = np.fft.rfftfreq(len(synth_chunk), 1.0 / sr_synth)
    mask_s = freqs_synth <= 5000
    fft_synth_db = 20 * np.log10(fft_synth[mask_s] + 1e-12)

    axes[1, 1].plot(freqs_synth[mask_s], fft_synth_db, color='#ff6600', linewidth=0.6)
    axes[1, 1].set_title('SYNTHESIS — Frequency Spectrum (Alto "ah")',
                          fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].set_ylim(-80, 10)
    axes[1, 1].grid(True, alpha=0.3)

    # ---- Figure 2: Overlaid spectra for direct comparison ----
    fig2, ax_overlay = plt.subplots(1, 1, figsize=(16, 6))
    fig2.suptitle("Spectral Overlay — Real vs Synthesis", fontsize=14, fontweight='bold')

    # Normalize both to 0dB peak for fair comparison
    real_norm = fft_real_db - np.max(fft_real_db)
    synth_norm = fft_synth_db - np.max(fft_synth_db)

    ax_overlay.plot(freqs_real[mask_r], real_norm, color='#00cc66',
                     linewidth=0.8, alpha=0.8, label='Real voice')
    ax_overlay.plot(freqs_synth[mask_s], synth_norm, color='#ff3366',
                     linewidth=0.8, alpha=0.8, label='Synthesis')
    ax_overlay.set_xlabel("Frequency (Hz)")
    ax_overlay.set_ylabel("dB (normalized)")
    ax_overlay.set_ylim(-60, 5)
    ax_overlay.legend(fontsize=12)
    ax_overlay.grid(True, alpha=0.3)

    # Save
    fig.savefig("voice_compare.png", dpi=150, bbox_inches='tight', facecolor='white')
    fig2.savefig("voice_overlay.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print("Saved: voice_compare.png")
    print("Saved: voice_overlay.png")


if __name__ == "__main__":
    analyze_comparison()
