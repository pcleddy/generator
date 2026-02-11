# Event Extraction & Score Generation Summary

## Overview
Successfully extracted event data from **8 CloudAutomat compositions** and generated both JSON event files and PNG graphical scores for use in an animated score player.

## Extraction Results

| Composition | Events | Duration | Seed | JSON File | Score PNG |
|---|---|---|---|---|---|
| Webern Pointillism | 32 | 50.0s | None | ✓ | ✓ |
| Berg Lyrical | 110 | 90.0s | None | ✓ | ✓ |
| Berg Extended (7 min) | 666 | 420.0s | 42 | ✓ | ✓ |
| Cage Ambient | 68 | 150.0s | 99 | ✓ | ✓ |
| Bells with Bergman Clock | 135 | 120.0s | 42 | ✓ | ✓ |
| Bells Pizzicato | 42 | 90.0s | 77 | ✓ | ✓ |
| Gentle Bells | 69 | 120.0s | 55 | ✓ | ✓ |
| Tubular Low | 18 | 70.0s | 31 | ✓ | ✓ |

**TOTAL: 1,140 events across 1,110 seconds (18.5 minutes)**

## Files Generated

### Event JSON Files
Located in `/sessions/sweet-nice-volta/mnt/generator/`:
- `events_webern_pointillism.json` (7.1 KB)
- `events_berg_lyrical.json` (23 KB)
- `events_berg_extended.json` (141 KB)
- `events_cage_ambient.json` (14 KB)
- `events_bells_bergman.json` (28 KB)
- `events_bells_pizz.json` (8.7 KB)
- `events_bells_gentle.json` (14 KB)
- `events_tubular_low.json` (3.8 KB)
- `events_manifest.json` (2.2 KB) - Master index of all compositions

### Graphical Score PNG Files
Located in `/sessions/sweet-nice-volta/mnt/generator/`:
- `webern_pointillism_score.png` (57 KB, 3200×1200)
- `berg_lyrical_score.png` (160 KB, 3200×1200)
- `berg_extended_score.png` (138 KB, 3200×1200)
- `cage_ambient_score.png` (76 KB, 3200×1200)
- `bells_bergman_score.png` (112 KB, 3200×1200)
- `bells_pizz_score.png` (68 KB, 3200×1200)
- `bells_gentle_score.png` (77 KB, 3200×1200)
- `tubular_low_score.png` (57 KB, 3200×1200)

## Event Data Structure

Each JSON file contains:
```json
{
  "title": "Composition Title",
  "duration": 90.0,
  "seed": 42,
  "audio_file": null,
  "n_events": 110,
  "events": [
    {
      "time": 1.5,           // start time in seconds
      "pc": 7,               // pitch class (0-11)
      "octave": 3,           // octave number
      "duration": 2.0,       // event duration in seconds
      "amplitude": 0.08,     // normalized amplitude (0-1)
      "type": "cello_pont",  // timbre/bell profile name
      "category": "lyrical"  // semantic category (pointillist/lyrical/bell/ambient/etc)
    },
    // ... more events
  ]
}
```

## Graphical Scores

Each PNG score visualization features:
- **Dark background** (#0a0a0f) for comfortable viewing
- **X-axis**: Time in seconds (0 to composition duration)
- **Y-axis**: Pitch in musical notation (C1 through C8)
- **Event markers**: Colored dots/shapes representing sound events
- **Colors by timbre type**:
  - Blue shades: Strings (cello pont/tasto)
  - Cyan: Flutes & winds
  - Amber/Orange: Bell instruments
  - Magenta: Pizzicato
  - Lavender: Woodwinds
- **Marker shapes by category**:
  - Diamond: Pointillist
  - Circle: Lyrical
  - Square: Extended
  - Hexagon: Bell-based
  - Triangle: Ambient/Interruptions
- **Event size**: Proportional to amplitude
- **Duration trails**: Horizontal lines extending from event (for long-duration tones)

## Extraction Logic

The extraction script (`extract_all_events.py`) performs:

1. **Function invocation**: Runs each composition's generation function with specified seed
2. **Event capture**: Extracts raw event dictionaries from function return values
3. **Normalization**: Standardizes event structure across all pieces:
   - Consolidates varied key names (e.g., `dur`/`duration`, `profile`/`timbre`, `amp`/`amplitude`)
   - Ensures consistent format for MIDI conversion
   - Adds semantic category labels
4. **JSON serialization**: Saves normalized events with metadata
5. **Visualization**: Renders PNG scores with matplotlib using:
   - Custom color palette per timbre type
   - Category-based marker shapes
   - Amplitude-based sizing
   - Duration-based line rendering

## Composition Notes

### Webern Pointillism
- 32 discrete sonic events separated by silence
- Timbral variety: cello (pont/tasto), flute, clarinet, bell, glass harmonica, oboe, pizzicato
- Sparse texture with ~60% silence

### Berg Lyrical
- 110 events forming continuous legato phrases
- Lush, romantic serialist approach
- 4 structural sections with dynamic arcs

### Berg Extended (7 minutes)
- 666 events across 420 seconds
- 7 cycles of material with evolving orchestration
- From emergence (pp) through storm (f) to dissolution (ppp)
- Most densely populated composition

### Cage Ambient
- 68 events: ambient drone foundation + prepared piano interruptions
- Ambiguous tonality with chance-based disruptions
- Includes cluster events, slaps, cascades, vocal elements

### Bells Suite (4 compositions)
- **Bergman Clock**: 135 events, layered glockenspiel, tubular bells, celesta
- **Pizz Variant**: 42 events, pizzicato patterns + heavy tubular bells
- **Gentle**: 69 events, soft wood xylophone + marimba + quiet bells
- **Tubular Low**: 18 events, deep tolling at slow tempo

## Usage for Animated Score Player

1. Load `events_manifest.json` to get list of all compositions
2. For each composition:
   - Load corresponding `events_{name}.json`
   - Parse events array
   - Map to visual parameters:
     - X: `time` (scaled to canvas width by composition `duration`)
     - Y: `pitch_to_midi(pc, octave)` → visual Y position
     - Color: by `type` field
     - Size: by `amplitude`
     - Duration: by `duration` field
3. Animate events sequentially by time value
4. Use PNG scores as reference visualization

## Technical Notes

- **Dependencies**: numpy, scipy, matplotlib
- **Sample rate**: 44100 Hz (used for some calculations)
- **Octave ranges**: Primarily octaves 2-6 (C2 to C7)
- **Amplitude range**: 0.02-0.40 normalized
- **Duration range**: 0.08s to 12.0s depending on composition style
- **Total computational time**: ~2 minutes for all extractions

## Files Location

All files are available at:
```
/sessions/sweet-nice-volta/mnt/generator/
```

Generated on: 2026-02-10
