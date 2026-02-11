# CloudAutomat Event Extraction & Score Generation

Complete event data extraction and graphical score visualization for all CloudAutomat compositions.

## Quick Start

**Load all compositions:**
```python
import json

# Load master index
with open('events_manifest.json', 'r') as f:
    manifest = json.load(f)

# Process each composition
for comp in manifest['compositions']:
    with open(comp['events_file'], 'r') as f:
        data = json.load(f)
    
    title = data['title']
    events = data['events']
    duration = data['duration']
    
    # Use in animated score player
    for event in events:
        time = event['time']
        pitch = event['pc']  # 0-11
        octave = event['octave']
        duration = event['duration']
        amplitude = event['amplitude']
        timbre_type = event['type']
        category = event['category']
```

## File Organization

```
/sessions/sweet-nice-volta/mnt/generator/
├── extract_all_events.py              # Main extraction script (544 lines)
├── events_manifest.json               # Master index of all compositions
├── events_webern_pointillism.json     # 32 events, 50s
├── events_berg_lyrical.json           # 110 events, 90s
├── events_berg_extended.json          # 666 events, 420s
├── events_cage_ambient.json           # 68 events, 150s
├── events_bells_bergman.json          # 135 events, 120s
├── events_bells_pizz.json             # 42 events, 90s
├── events_bells_gentle.json           # 69 events, 120s
├── events_tubular_low.json            # 18 events, 70s
├── webern_pointillism_score.png       # Graphical score
├── berg_lyrical_score.png
├── berg_extended_score.png
├── cage_ambient_score.png
├── bells_bergman_score.png
├── bells_pizz_score.png
├── bells_gentle_score.png
├── tubular_low_score.png
├── README_EXTRACTION.md               # This file
├── EXTRACTION_SUMMARY.md              # Detailed summary
└── EVENT_DATA_VERIFICATION.txt        # Technical details
```

## Data Structure

### Manifest (events_manifest.json)
```json
{
  "compositions": [
    {
      "composition": "webern_pointillism",
      "title": "Webern Pointillism",
      "events_file": "events_webern_pointillism.json",
      "score_file": "webern_pointillism_score.png",
      "n_events": 32,
      "duration": 50.0,
      "seed": null
    },
    // ... 7 more compositions
  ],
  "total_events": 1140,
  "total_duration": 1110.0,
  "generated": "2026-02-10",
  "description": "Event data extracted from all CloudAutomat compositions..."
}
```

### Event Object (per event in events array)
```json
{
  "time": 2.786,              // Start time in seconds
  "pc": 0,                    // Pitch class (0-11: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
  "octave": 3,                // Octave number (typically 2-6)
  "duration": 0.180,          // Duration in seconds
  "amplitude": 0.023,         // Normalized amplitude (0.0-1.0)
  "type": "oboe_pp",          // Timbre/profile name
  "category": "pointillist"   // Semantic category
}
```

## Compositions Overview

| # | Composition | Style | Events | Duration | Seed | Density |
|---|---|---|---|---|---|---|
| 1 | Webern Pointillism | Sparse, isolated | 32 | 50s | None | 0.64/s |
| 2 | Berg Lyrical | Dense, legato | 110 | 90s | None | 1.22/s |
| 3 | Berg Extended | Extended form | 666 | 420s | 42 | 1.59/s |
| 4 | Cage Ambient | Ambient + prepared | 68 | 150s | 99 | 0.45/s |
| 5 | Bells Bergman | Layered bells | 135 | 120s | 42 | 1.13/s |
| 6 | Bells Pizz | Pizz + bells | 42 | 90s | 77 | 0.47/s |
| 7 | Gentle Bells | Soft wood + bells | 69 | 120s | 55 | 0.58/s |
| 8 | Tubular Low | Deep tolling | 18 | 70s | 31 | 0.26/s |

## Instrument/Timbre Types

### Strings
- `cello_pont` - sul ponticello (bright, glassy)
- `cello_tasto` - sul tasto (dark, muffled)
- `pizzicato` - plucked strings

### Winds
- `flute_breathy` - breathy flute
- `clarinet_chalumeau` - low clarinet (hollow)
- `oboe_pp` - quiet oboe

### Bells & Percussion
- `glockenspiel` - bright, high, quick decay
- `celesta` - soft, crystalline
- `tubular_bell` - deep, churchy, long ring
- `church_bell` - heavy, massive
- `wind_chime` - sparkling, high
- `bell_struck` - generic struck bell

### Wood & Mallets
- `wood_xylophone` - rosewood bars (warm)
- `marimba` - deeper, warmer cousin
- `glass_harmonica` - glassy, ethereal

### Special/Ambient
- `ambient_primary` - drone foundation
- `ambient_inverted` - inverted drone
- `ambient_drone` - sub-bass drone
- Prepared piano types: clusters, slaps, cascades, vocal screams

## Event Categories

- **pointillist** - Discrete events with silence (Webern style)
- **lyrical** - Continuous legato phrases (Berg style)
- **extended** - Extended form with cycles (Berg 7-min)
- **bell** - Percussion/bell-based
- **ambient** - Drone/background textures
- **interruption** - Disruptive events (prepared piano)

## Graphical Scores

Each PNG visualization includes:

- **Background**: Dark (#0a0a0f) for comfortable viewing
- **X-axis**: Time in seconds (0 to composition duration)
- **Y-axis**: Pitch in semitones (C1-C8, labeled with note names)
- **Events**: Color-coded dots/shapes representing sound events
- **Size**: Proportional to amplitude (louder = bigger)
- **Duration trails**: Lines extending from events for long notes
- **Markers**: Different shapes per category
  - Diamond (◇): Pointillist
  - Circle (●): Lyrical
  - Square (■): Extended
  - Hexagon (⬡): Bell/Percussion
  - Triangle (▲): Ambient
  - Inverted Triangle (▼): Interruption

### Color Scheme
- **Deep Blue** (#1a5276-#4488dd): Strings
- **Cyan** (#4499cc-#66ddff): Flutes & winds
- **Amber** (#ffaa44): Bells & warm
- **Magenta** (#ff66aa): Pizzicato
- **Lavender** (#ccaaff-#ddaaff): Woodwinds
- **Orange** (#ff6600-#ff6644): Prepared piano
- **Yellow** (#ffcc00): Vocal/extreme
- **Green** (#aaffcc): Wind chimes

## Usage Examples

### Load and Process All Events
```python
import json

with open('events_manifest.json', 'r') as f:
    manifest = json.load(f)

for comp_info in manifest['compositions']:
    with open(comp_info['events_file']) as f:
        data = json.load(f)
    
    print(f"{data['title']}: {len(data['events'])} events in {data['duration']}s")
```

### Parse Single Event for Visualization
```python
import json

with open('events_webern_pointillism.json') as f:
    data = json.load(f)

for event in data['events']:
    # Convert to visualization coordinates
    x = event['time']  # Seconds on timeline
    
    # Convert pitch class + octave to MIDI note
    midi = event['pc'] + (event['octave'] + 1) * 12
    y = midi  # Y coordinate
    
    # Size proportional to amplitude
    size = event['amplitude'] * 1000
    
    # Color/marker by type and category
    timbre = event['type']
    style = event['category']
    
    # Use duration for note-off timing or bar length
    duration = event['duration']
```

### Create Animation Timeline
```python
import json

with open('events_berg_extended.json') as f:
    data = json.load(f)

# Sort events by time
events = sorted(data['events'], key=lambda e: e['time'])

# Animate at 60 FPS
fps = 60
frame_duration = 1.0 / fps

for event in events:
    # Render event at time
    onset_frame = int(event['time'] * fps)
    offset_frame = int((event['time'] + event['duration']) * fps)
    
    for frame in range(onset_frame, offset_frame):
        # Render event at current frame
        # Use amplitude for fade envelope
        progress = (frame - onset_frame) / (offset_frame - onset_frame)
        alpha = 1.0 - progress  # Fade out
        render_event(event, alpha)
```

## Statistics

- **Total Compositions**: 8
- **Total Events**: 1,140
- **Total Duration**: 1,110 seconds (18.5 minutes)
- **Average Events/Second**: 1.03
- **Pitch Range**: C2 (MIDI 24) to C7 (MIDI 84)
- **Amplitude Range**: 0.02 to 0.40 (normalized)
- **Event Duration Range**: 0.08s to 12.0s

- **JSON Data**: 273 KB
- **PNG Scores**: 802 KB
- **Total**: ~1.1 MB

## Technical Notes

- **Extraction Method**: Direct function invocation with canonical seeds
- **Normalization**: Standardized event structure across all pieces
- **Serialization**: JSON UTF-8 with 2-space indentation
- **Visualization**: Matplotlib with dark theme, 100 DPI
- **Dependencies**: numpy, scipy, matplotlib

## Running the Extraction Script

To regenerate all data:

```bash
cd /sessions/sweet-nice-volta/mnt/generator
python extract_all_events.py
```

Output: All JSON files and PNG scores will be regenerated.

## Integration with Animated Score Player

1. **Discovery**: Load `events_manifest.json`
2. **Load**: Load corresponding `events_{name}.json` for each piece
3. **Parse**: Extract events array
4. **Visualize**: Map events to canvas coordinates:
   - X: `time / duration * canvas_width`
   - Y: `(midi_note / 127) * canvas_height`
   - Color: `type → color_lookup`
   - Size: `amplitude * marker_size`
5. **Animate**: Play events sequentially by time value
6. **Reference**: Use PNG scores as visual guide

---

**Generated**: 2026-02-10  
**Status**: Complete and verified  
**Ready for production use**
