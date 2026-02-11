"""
fm_brass.py — FM synthesis for brass instruments.

Brass instruments have a distinctive characteristic: brightness correlates
with loudness. The "blat" of a trumpet attack is high-frequency content
that decays faster than the fundamental. FM synthesis models this naturally
through a time-varying modulation index.

Core idea:
  output = sin(2π·fc·t + MI(t) · sin(2π·fm·t))

Where MI(t) starts high (bright attack) and decays to a lower sustain value.
The carrier:modulator ratio is 1:1 for all standard brass, giving a fully
harmonic spectrum. The modulation index controls how many sidebands
(harmonics) are present — higher MI = more harmonics = brighter.

DX7-inspired: two operator pairs summed for richness, with optional
feedback on the modulators.
"""

import numpy as np
from ..config import SAMPLE_RATE


def fm_brass_tone(t, start, freq, duration, amplitude, profile, rng):
    """Synthesize a brass tone using 2-operator FM synthesis.

    Args:
        t: global time array
        start: note onset time (seconds)
        freq: fundamental frequency (Hz)
        duration: note duration (seconds)
        amplitude: 0-1 amplitude
        profile: dict with FM brass parameters (see FM_BRASS_PROFILES)
        rng: SeedManager instance

    Returns:
        numpy array same length as t, with the brass tone mixed in
    """
    n_samples = len(t)
    voice = np.zeros(n_samples)

    if freq < 30 or freq > 4000:
        return voice

    mask = (t >= start) & (t < start + duration)
    n_active = int(np.sum(mask))
    if n_active == 0:
        return voice

    t_local = t[mask] - start
    t_norm = t_local / max(duration, 1e-6)

    # ── Profile parameters ──
    mod_ratio = profile.get("mod_ratio", 1.0)
    peak_mi = profile.get("peak_mi", 10.0)
    sustain_mi = profile.get("sustain_mi", 3.5)
    mi_attack = profile.get("mi_attack", 0.08)    # seconds to peak MI
    mi_decay = profile.get("mi_decay", 0.4)       # seconds from peak to sustain MI
    amp_attack = profile.get("amp_attack", 0.05)   # amplitude envelope attack
    amp_decay = profile.get("amp_decay", 0.3)      # to sustain level
    amp_sustain = profile.get("amp_sustain", 0.7)   # sustain level (0-1)
    feedback = profile.get("feedback", 0.0)         # modulator self-feedback
    noise_amount = profile.get("noise_amount", 0.08)
    noise_decay = profile.get("noise_decay", 0.3)   # breath noise decay time
    vibrato_rate = profile.get("vibrato_rate", 5.0)
    vibrato_depth = profile.get("vibrato_depth", 8)  # cents
    vibrato_delay = profile.get("vibrato_delay", 0.3)  # seconds before vibrato onset
    second_op_detune = profile.get("second_op_detune", 0.002)  # slight detune for 2nd pair
    second_op_mix = profile.get("second_op_mix", 0.3)  # mix level of 2nd operator pair

    # ── Frequencies ──
    fc = freq
    fm = freq * mod_ratio

    # ── Pitch micro-drift (natural player intonation) ──
    n_drift = max(int(duration * 15), 4)
    drift_walk = np.cumsum(rng.randn(n_drift) * 0.2)
    drift_walk = np.clip(drift_walk, -4, 4)  # ±4 cents max
    drift_cents = np.interp(
        np.linspace(0, 1, n_active),
        np.linspace(0, 1, n_drift),
        drift_walk
    )
    drift_mult = 2 ** (drift_cents / 1200)

    # ── Vibrato (delayed onset, like a real brass player) ──
    if vibrato_delay < duration and vibrato_depth > 0:
        vib_onset = np.clip((t_local - vibrato_delay) / 0.25, 0, 1)
        # Slight rate wobble for naturalness
        rate_mod = 1.0 + 0.04 * np.sin(2 * np.pi * 0.25 * t_local)
        vibrato = vib_onset * vibrato_depth * np.sin(
            2 * np.pi * vibrato_rate * rate_mod * t_local
        )
        vib_mult = 2 ** (vibrato / 1200)
    else:
        vib_mult = 1.0

    freq_mod = drift_mult * vib_mult

    # ── Modulation Index envelope ──
    # Fast attack to peak_mi, then decay to sustain_mi
    mi_env = np.ones(n_active) * sustain_mi

    # Attack ramp
    mi_att_samples = max(int(mi_attack * SAMPLE_RATE), 1)
    if mi_att_samples < n_active:
        # Exponential attack (fast rise)
        att_curve = 1.0 - np.exp(-5.0 * np.linspace(0, 1, mi_att_samples))
        mi_env[:mi_att_samples] = att_curve * peak_mi

    # Decay from peak to sustain
    mi_dec_start = mi_att_samples
    mi_dec_samples = max(int(mi_decay * SAMPLE_RATE), 1)
    mi_dec_end = min(mi_dec_start + mi_dec_samples, n_active)
    if mi_dec_start < n_active:
        decay_len = mi_dec_end - mi_dec_start
        if decay_len > 0:
            # Exponential decay from peak_mi to sustain_mi
            decay_curve = np.exp(-4.0 * np.linspace(0, 1, decay_len))
            mi_env[mi_dec_start:mi_dec_end] = sustain_mi + (peak_mi - sustain_mi) * decay_curve

    # ── Amplitude envelope ──
    amp_env = np.ones(n_active)

    # Attack
    att_samples = max(int(amp_attack * SAMPLE_RATE), 1)
    if att_samples < n_active:
        amp_env[:att_samples] = 1.0 - np.exp(-4.0 * np.linspace(0, 1, att_samples))

    # Decay to sustain
    dec_start = att_samples
    dec_samples = max(int(amp_decay * SAMPLE_RATE), 1)
    dec_end = min(dec_start + dec_samples, n_active)
    if dec_start < n_active and dec_end > dec_start:
        dec_len = dec_end - dec_start
        amp_env[dec_start:dec_end] = amp_sustain + (1.0 - amp_sustain) * np.exp(
            -3.0 * np.linspace(0, 1, dec_len)
        )
        if dec_end < n_active:
            amp_env[dec_end:] = amp_sustain

    # Final release — gentle tail
    release_len = min(int(0.08 * SAMPLE_RATE), n_active // 4)
    if release_len > 1:
        amp_env[-release_len:] *= np.linspace(1, 0, release_len)

    # Anti-click at very start
    click_guard = min(int(0.003 * SAMPLE_RATE), n_active)
    if click_guard > 0:
        amp_env[:click_guard] *= np.linspace(0, 1, click_guard)

    # ── FM Synthesis — Operator Pair 1 ──
    # Phase accumulation (more accurate than freq * t for varying frequency)
    inst_fm = fm * freq_mod
    inst_fc = fc * freq_mod

    # Modulator phase
    phase_m1 = np.cumsum(2 * np.pi * inst_fm / SAMPLE_RATE)
    phase_m1 += rng.uniform(0, 2 * np.pi)

    # Modulator with optional self-feedback
    if feedback > 0.01:
        # Feedback FM: modulator feeds back into itself
        # Must be done sample-by-sample for accuracy, but we can approximate
        # with a single-sample delay approach
        modulator1 = np.zeros(n_active)
        prev_mod = 0.0
        for i in range(n_active):
            modulator1[i] = np.sin(phase_m1[i] + feedback * prev_mod)
            prev_mod = modulator1[i]
    else:
        modulator1 = np.sin(phase_m1)

    # Carrier phase with FM
    carrier_phase1 = np.cumsum(
        2 * np.pi * inst_fc / SAMPLE_RATE
    ) + mi_env * modulator1
    carrier_phase1 += rng.uniform(0, 2 * np.pi)
    signal1 = np.sin(carrier_phase1)

    # ── FM Synthesis — Operator Pair 2 (richness) ──
    # Slightly detuned second pair for ensemble width
    if second_op_mix > 0.01:
        fm2 = fm * (1.0 + second_op_detune)
        fc2 = fc * (1.0 + second_op_detune * 0.5)

        inst_fm2 = fm2 * freq_mod
        inst_fc2 = fc2 * freq_mod

        phase_m2 = np.cumsum(2 * np.pi * inst_fm2 / SAMPLE_RATE)
        phase_m2 += rng.uniform(0, 2 * np.pi)

        # Second modulator — slightly different MI for variation
        modulator2 = np.sin(phase_m2)
        mi_env2 = mi_env * 0.85  # slightly less bright

        carrier_phase2 = np.cumsum(
            2 * np.pi * inst_fc2 / SAMPLE_RATE
        ) + mi_env2 * modulator2
        carrier_phase2 += rng.uniform(0, 2 * np.pi)
        signal2 = np.sin(carrier_phase2)

        # Mix both operator pairs
        signal = signal1 + second_op_mix * signal2
        signal /= (1.0 + second_op_mix)
    else:
        signal = signal1

    # ── Breath noise ──
    if noise_amount > 0.001:
        noise = rng.randn(n_active)

        # Bandpass around 2-5 kHz for breath character
        from scipy.signal import lfilter
        center = 3500
        w0 = 2 * np.pi * center / SAMPLE_RATE
        Q = 1.5
        alpha = np.sin(w0) / (2 * Q)
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
        noise = lfilter(b, a, noise)

        # Noise envelope: peaks at attack, decays fast
        noise_env = np.zeros(n_active)
        noise_att = max(int(mi_attack * SAMPLE_RATE), 1)
        if noise_att < n_active:
            noise_env[:noise_att] = np.linspace(0, 1, noise_att)
        noise_dec_samples = max(int(noise_decay * SAMPLE_RATE), 1)
        noise_dec_end = min(noise_att + noise_dec_samples, n_active)
        if noise_att < n_active:
            dec_len = noise_dec_end - noise_att
            if dec_len > 0:
                noise_env[noise_att:noise_dec_end] = np.exp(
                    -5.0 * np.linspace(0, 1, dec_len)
                )
            # Tiny residual breath during sustain
            if noise_dec_end < n_active:
                noise_env[noise_dec_end:] = 0.05

        signal += noise * noise_env * noise_amount

    # ── Apply amplitude envelope ──
    voice[mask] = signal * amp_env * amplitude

    # Normalize within the active region
    active_peak = np.max(np.abs(voice[mask]))
    if active_peak > 0:
        voice[mask] = voice[mask] / active_peak * amplitude

    return voice
