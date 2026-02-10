# Ambient Synthesis Project: Policy & Procedure

**CloudAutomat Labs**  
**Project:** Impossible Sound Generator  
**Version:** 0.2.0  
**Last Updated:** February 10, 2026

---

## 1. Project Scope

Generate beatless, ambient soundscapes that ride the boundary between 12-tone serialist order and microtonal chaos. No rhythm, minimal resonance, slow evolution. Target: 60-120 second pieces that feel "impossible"—simultaneously structured and toneless.

---

## 2. Core Architecture

### Synthesis Layers
1. **Primary 12-tone row** (ORDER): Slow march through serial pitch material
2. **Inverted row** (ORDER): Mirrored structure, staggered offset
3. **Microtonal chaos** (CHAOS): Detuned drifters; 40% anchored to 12-tone, 60% pure chaos
4. **Sub-bass foundation** (RESONANCE): Very slow, barely perceptible LFO modulation
5. **Microtonal clusters** (ORDER/CHAOS collision): Small chords of detuned voices around 12-tone pitch centers

---

## 3. Code Standards

### 3.1 Pitch Material
- Base frequency: C3 = 130.81 Hz
- 12-tone row defined as array: `[0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10]`
- Microtonal detuning in **cents** (1 semitone = 100 cents)
- Range for anchored chaos: ±30 cents; pure chaos: ±50 cents

### 3.2 Voice Function Signature
```python
harmonic_voice(start_time, pitch_semitones, octave, duration, fade_in, fade_out, 
               harmonic_mix=None, amplitude=0.08, mod_freq=None, mod_depth=0)
```

**Parameters:**
- `start_time`: seconds
- `pitch_semitones`: float (can be microtonal)
- `octave`: integer shift from C3 (-2 to +1 typical)
- `duration`: seconds
- `fade_in/out`: seconds (no hard edges)
- `harmonic_mix`: list of amplitudes `[fund, 2nd, 3rd...]` (normalized)
- `amplitude`: overall output level (0-1)
- `mod_freq`: LFO frequency in Hz (0.02-0.2 typical)
- `mod_depth`: pitch wobble in cents

### 3.3 Layer Density Rules
- Primary row: One pitch every 5.5 seconds
- Inverted row: Offset by 3 seconds, same rhythm
- Microtonal chaos: 24 voices distributed randomly across duration
- Clusters: Placed every 18 seconds, 2-4 voices per cluster

---

## 4. Tuning Parameters (Current)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 44100 Hz | CD quality; sufficient for <10kHz content |
| Duration | 120 seconds | Allows evolution; multi-layer development |
| Primary amplitude | 0.07 | Loud enough to anchor structure |
| Chaos amplitude | 0.04 | Lower; texture role |
| Sub-bass amplitude | 0.035 | Nearly subliminal; pressure only |
| Cluster amplitude | 0.045 | Between chaos and primary |
| Fade in (order) | 2-2.5s | Slow emergence |
| Fade out (order) | 2-2.5s | Gradual dissolution |
| Fade in (chaos) | 3s | Softer entry |
| Fade out (chaos) | 4s | Longer tail |
| Harmonic mix | [0.85-0.9, 0.1-0.2] | Subtle shimmer without presence |
| LFO mod range | 0.05-0.2 Hz | Very slow; barely perceptible drift |

---

## 5. Modification Workflow

### 5.1 Before Changing Code
1. Branch: `git checkout -b experiment/[descriptor]`
2. Create backup: `cp ambient_synthesis.py ambient_synthesis.backup.py`
3. Document hypothesis: Comment or README note

### 5.2 Parameter Tweaking
Changes to tune:
- Comment out old value
- Add new value with inline reason
- Example:
```python
# Original: amplitude=0.07
# Test: More prominent primary layer
audio += harmonic_voice(..., amplitude=0.09)  # TEMP: testing presence
```

### 5.3 Testing & Listening
1. Generate: `python ambient_synthesis.py`
2. Listen: At least 30 seconds (minimum context)
3. Evaluate against: Order/chaos balance, resonance character, evolution
4. If good: Commit with message explaining change
5. If not: Revert and try different parameter

---

## 6. Output Standards

### 6.1 File Format
- **Format**: WAV (uncompressed)
- **Bit depth**: 16-bit PCM
- **Sample rate**: 44100 Hz
- **Naming**: `ambient_XX.wav` (version number increments)

### 6.2 Audio Quality
- Normalize to -3dB peak (leave headroom)
- Soft compression via tanh (prevents clipping, adds warmth)
- No limiting or hard clipping
- Check for clicks/pops at layer boundaries (should be smooth via envelopes)

---

## 7. Documentation

### 7.1 Metadata per Version
After generation, update version log:
```markdown
## [0.2.0] - 2026-02-10
- Dual 12-tone rows (primary + inversion)
- 24-voice microtonal chaos layer
- Sub-bass with slow LFO modulation
- Cluster collision points every 18 seconds
- Duration: 120s
- Tuning: Order/chaos balance = 60/40
```

### 7.2 Listening Notes
Format: Describe what you hear and why it works or doesn't.
```
### ambient_02.wav
**Strong points:**
- Inversion creates nice shadow; feels intentional
- Cluster collisions land at organic moments

**Issues:**
- Sub-bass barely audible; consider raising to 0.04
- Chaos layer sometimes overshadows order (consider 35/65 split)

**Next iteration:**
- Increase LFO depth slightly
- Add fifth cluster layer with different pitch center
```

---

## 8. Experimental Branches

### 8.1 Active Explorations
| Branch | Objective | Status |
|--------|-----------|--------|
| `main` | Stable baseline | Active |
| `experiment/higher-density` | More voices, tighter clustering | In progress |
| `experiment/fifth-layer` | Additional 12-tone variant | Planned |

### 8.2 Archival
Completed experiments tagged: `v0.1.0`, `v0.2.0`, etc.  
Keep winning variations; delete obviously failed branches after 1 month.

---

## 9. Cowork Integration (Future)

### 9.1 Batch Generation
Script parameter: Seed variation
```bash
python ambient_synthesis.py --seed 42 --output ambient_seed42.wav
python ambient_synthesis.py --seed 100 --output ambient_seed100.wav
```

### 9.2 Metadata Export
Auto-generate JSON for each output:
```json
{
  "version": "0.2.0",
  "seed": 42,
  "duration": 120,
  "layers": 5,
  "order_chaos_ratio": "60/40",
  "timestamp": "2026-02-10T14:30:00Z"
}
```

---

## 10. Review & Evolution

**Next milestones:**
- [ ] Test cluster density (add/remove clusters)
- [ ] Experiment with transposed 12-tone rows
- [ ] Add spectral analysis (visual feedback on balance)
- [ ] Record listening session notes; formalize feedback loop

**Review cycle:** Weekly; listen critically, adjust parameters, commit good versions.

---

**Maintainer:** Paul (CloudAutomat Labs)  
**Last Generated:** ambient_02.wav (2026-02-10)
