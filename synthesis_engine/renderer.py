"""
renderer.py — Events → audio → WAV/MP3 + JSON + score PNG.

The rendering pipeline. Takes a list of SynthEvents, synthesizes each
one using the appropriate tone generator, mixes, applies reverb, normalizes.
"""

import numpy as np
from scipy.io import wavfile
import subprocess
import json
import os

from .config import SAMPLE_RATE, freq_from_pitch_class
from .registry import InstrumentRegistry
from .synthesis.tone import pointillist_tone
from .synthesis.bell import bell_strike, wood_strike
from .synthesis.plucked import karplus_strong
from .synthesis.fm_brass import fm_brass_tone
from .synthesis.reverb import simple_reverb


class Renderer:
    """Render SynthEvents to audio.

    Usage:
        renderer = Renderer()
        audio = renderer.render(events, duration=30.0)
        renderer.save_wav(audio, "output.wav")
        renderer.save_mp3("output.wav", "output.mp3")
    """

    def __init__(self, sample_rate=SAMPLE_RATE, registry=None):
        self.sample_rate = sample_rate
        self.registry = registry or InstrumentRegistry()

    def render(self, events, duration, rng=None, reverb_preset="room"):
        """Synthesize all events into a mono audio array.

        Args:
            events: list of SynthEvent
            duration: total duration in seconds
            rng: SeedManager for any synthesis-level randomness
            reverb_preset: name of reverb preset or None for dry

        Returns:
            numpy array of audio samples (float64, -1 to 1)
        """
        from .seed_manager import SeedManager
        if rng is None:
            rng = SeedManager(42)

        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = np.zeros(n_samples)

        for event in events:
            freq = freq_from_pitch_class(event.pitch_class, event.octave)
            instrument = event.instrument
            category = self._get_category(instrument)

            if category == "bell":
                profile = self._get_bell_profile(instrument)
                audio += bell_strike(
                    t, event.time, freq, event.duration,
                    event.amplitude, profile, rng
                )
            elif category == "wood":
                profile = self._get_wood_profile(instrument)
                audio += wood_strike(
                    t, event.time, freq, event.duration,
                    event.amplitude, profile, rng
                )
            elif category == "plucked":
                profile = self._get_plucked_profile(instrument)
                audio += karplus_strong(
                    t, event.time, freq, event.duration,
                    event.amplitude, profile, rng
                )
            elif category == "fm_brass":
                profile = self._get_fm_brass_profile(instrument)
                audio += fm_brass_tone(
                    t, event.time, freq, event.duration,
                    event.amplitude, profile, rng
                )
            else:
                # Default: pointillist_tone (strings, winds, etc.)
                timbre = self._get_timbre(instrument)
                audio += pointillist_tone(
                    t, event.time, event.pitch_class, event.octave,
                    event.duration, event.amplitude, timbre, rng
                )

        # Apply reverb
        if reverb_preset:
            audio = simple_reverb(audio, preset=reverb_preset,
                                 sample_rate=self.sample_rate)

        # Gentle fade at end
        fade_len = min(int(2.0 * self.sample_rate), len(audio) // 4)
        if fade_len > 0:
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.92

        return audio

    def save_wav(self, audio, path):
        """Write audio array to WAV file."""
        audio_16 = np.int16(audio * 32767)
        wavfile.write(path, self.sample_rate, audio_16)
        return path

    def save_mp3(self, wav_path, mp3_path=None, bitrate='192k'):
        """Convert WAV to MP3 using ffmpeg."""
        if mp3_path is None:
            mp3_path = wav_path.replace('.wav', '.mp3')
        subprocess.run(
            ['ffmpeg', '-y', '-i', wav_path,
             '-codec:a', 'libmp3lame', '-b:a', bitrate, mp3_path],
            capture_output=True
        )
        return mp3_path

    def render_and_save(self, composition, output_prefix, reverb_preset=None):
        """Full pipeline: render → WAV → MP3 → JSON.

        Args:
            composition: Composition instance (must have called .generate())
            output_prefix: e.g. "cage_bells_family" → creates _01.wav, _01.mp3, .json

        Returns:
            dict with file paths
        """
        events = composition.sorted_events()
        preset = reverb_preset or composition.reverb_preset

        print(f"Rendering {composition.title}...")
        print(f"  Events: {len(events)}, Duration: {composition.duration}s, "
              f"Reverb: {preset}")

        audio = self.render(
            events, composition.duration,
            rng=composition.rng, reverb_preset=preset
        )

        wav_path = f"{output_prefix}_01.wav"
        mp3_path = f"{output_prefix}_01.mp3"
        json_path = f"{output_prefix}.json"

        self.save_wav(audio, wav_path)
        print(f"  Written: {wav_path}")

        self.save_mp3(wav_path, mp3_path)
        print(f"  Written: {mp3_path}")

        composition.save_json(json_path, audio_file=os.path.basename(mp3_path))
        print(f"  Written: {json_path}")

        print(f"\n{composition.summary()}")

        return {
            'wav': wav_path,
            'mp3': mp3_path,
            'json': json_path,
            'audio': audio,
        }

    def _get_category(self, instrument_name):
        """Determine instrument category."""
        if self.registry.has(instrument_name):
            profile = self.registry.get(instrument_name)
            return profile.get('category', 'string')
        # Fallback heuristics
        if 'bell' in instrument_name or instrument_name in ('glockenspiel', 'celesta', 'wind_chime'):
            return 'bell'
        if 'wood' in instrument_name or instrument_name in ('marimba',):
            return 'wood'
        if instrument_name.startswith('fm_'):
            return 'fm_brass'
        if instrument_name in ('guitar_nylon', 'guitar_steel', 'harp', 'plucked_cello',
                               'banjo', 'sitar', 'koto', 'harpsichord'):
            return 'plucked'
        return 'string'

    def _get_bell_profile(self, name):
        """Get bell profile dict."""
        if self.registry.has(name):
            return self.registry.get(name)
        from .profiles.bells import BELL_PROFILES
        return BELL_PROFILES.get(name, BELL_PROFILES['tubular_bell'])

    def _get_wood_profile(self, name):
        """Get wood percussion profile dict."""
        if self.registry.has(name):
            return self.registry.get(name)
        from .profiles.wood import WOOD_PROFILES
        return WOOD_PROFILES.get(name, WOOD_PROFILES['wood_xylophone'])

    def _get_plucked_profile(self, name):
        """Get plucked string (Karplus-Strong) profile dict."""
        if self.registry.has(name):
            return self.registry.get(name)
        from .profiles.plucked import PLUCKED_PROFILES
        return PLUCKED_PROFILES.get(name, PLUCKED_PROFILES['guitar_nylon'])

    def _get_fm_brass_profile(self, name):
        """Get FM brass profile dict."""
        if self.registry.has(name):
            return self.registry.get(name)
        from .profiles.fm_brass import FM_BRASS_PROFILES
        return FM_BRASS_PROFILES.get(name, FM_BRASS_PROFILES['fm_trumpet'])

    def _get_timbre(self, name):
        """Get timbre dict for pointillist_tone."""
        if self.registry.has(name):
            return self.registry.get(name)
        from .profiles.strings import TIMBRES
        return TIMBRES.get(name, TIMBRES['cello_pont'])
