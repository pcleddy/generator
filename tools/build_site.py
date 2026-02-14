#!/usr/bin/env python3
"""
build_site.py — Generate index.html and player.html from pieces.json + templates.

Single source of truth: pieces.json defines all compositions.
Templates have placeholder markers that get replaced with generated HTML/JS.

Usage:
    python tools/build_site.py          # from repo root
    python tools/build_site.py --check  # verify output matches current files (no write)
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIECES_JSON = os.path.join(REPO_ROOT, "pieces.json")
INDEX_TEMPLATE = os.path.join(REPO_ROOT, "templates", "index_template.html")
PLAYER_TEMPLATE = os.path.join(REPO_ROOT, "templates", "player_template.html")
INDEX_OUTPUT = os.path.join(REPO_ROOT, "index.html")
PLAYER_OUTPUT = os.path.join(REPO_ROOT, "player.html")

GITHUB_BASE = "https://github.com/pcleddy/generator/blob/main"


def load_pieces():
    with open(PIECES_JSON, "r") as f:
        return json.load(f)


def generate_piece_card(piece):
    """Generate one piece-card div from a piece dict."""
    # Tags
    tags_html = "\n".join(
        f'    <span class="tag {t["class"]}">{t["label"]}</span>'
        for t in piece["tags"]
    )

    # Score section (optional)
    score_html = ""
    if piece.get("score_png"):
        score_html = f"""
  <div class="score-section">
    <a href="{piece['score_png']}" target="_blank">
      <img class="score-img" src="{piece['score_png']}"
           alt="{piece['title']} — Graphical Score" loading="lazy">
    </a>
    <div class="score-caption">
      {piece['score_caption']}
    </div>
  </div>"""

    return f"""<div class="piece-card">
  <div class="piece-header">
    <div class="piece-title">{piece['title']}</div>
    <div class="piece-meta">{piece['meta']}</div>
  </div>
  <div class="piece-description">
    {piece['description']}
  </div>
  <div class="piece-tags">
{tags_html}
  </div>
  <audio controls preload="none">
    <source src="{piece['audio_url']}" type="audio/mpeg">
  </audio>{score_html}
</div>"""


def generate_source_files(pieces):
    """Generate the source files grid from pieces."""
    items = []

    # Add source files for each piece that has one
    seen = set()
    for p in pieces:
        sf = p.get("source_file")
        if sf and sf not in seen:
            seen.add(sf)
            filename = os.path.basename(sf)
            desc = p["title"].split(" — ")[0] if " — " in p["title"] else p["title"]
            items.append(
                f'  <div class="file-item">\n'
                f'    <a href="{GITHUB_BASE}/{sf}">{filename}</a>\n'
                f'    <div class="file-desc">{desc}</div>\n'
                f'  </div>'
            )

    # Add standard tool files
    tool_files = [
        ("tools/extract_all_events.py", "Universal score + event extractor"),
        ("tools/score_generator.py", "Graphical + text score generation"),
        ("tools/sound_demo.py", "Timbre gallery test bench"),
        ("tools/voice_demo.py", "Vocal synthesis test bench"),
        ("tools/analyze_paul.py", "LPC voice analysis"),
        ("tools/build_site.py", "JSON-driven site generator"),
        ("player.html", "Animated score player (Canvas)"),
    ]
    for path, desc in tool_files:
        if path not in seen:
            seen.add(path)
            filename = os.path.basename(path)
            items.append(
                f'  <div class="file-item">\n'
                f'    <a href="{GITHUB_BASE}/{path}">{filename}</a>\n'
                f'    <div class="file-desc">{desc}</div>\n'
                f'  </div>'
            )

    return '<div class="file-list">\n' + "\n".join(items) + "\n</div>"


def generate_dropdown_options(pieces):
    """Generate <option> tags for player dropdown."""
    lines = []
    for p in pieces:
        if p.get("has_score_json") and p.get("player_label"):
            lines.append(
                f'                    <option value="{p["id"]}">{p["player_label"]}</option>'
            )
    return "\n".join(lines)


def generate_config_pieces(pieces):
    """Generate CONFIG.pieces JS entries."""
    lines = []
    for p in pieces:
        if p.get("has_score_json") and p.get("player_name"):
            lines.append(f"                {p['id']}: '{p['player_name']}',")
    return "\n".join(lines)


def build_index(pieces, template):
    """Build index.html from template + pieces."""
    # Generate all piece cards
    cards = "\n\n".join(generate_piece_card(p) for p in pieces)

    # Generate source files section
    source_files = generate_source_files(pieces)

    # Replace markers
    html = template.replace("<!-- PIECES_GO_HERE -->", cards)
    html = html.replace("<!-- SOURCE_FILES_GO_HERE -->", source_files)

    return html


def build_player(pieces, template):
    """Build player.html from template + pieces."""
    # Generate dropdown options
    options = generate_dropdown_options(pieces)

    # Generate config pieces
    config = generate_config_pieces(pieces)

    # Replace markers
    html = template.replace("<!-- DROPDOWN_OPTIONS -->", options)
    html = template if "<!-- DROPDOWN_OPTIONS -->" not in template else html
    html = html.replace("/* CONFIG_PIECES */", config)

    return html


def main():
    check_mode = "--check" in sys.argv

    print("build_site.py — Generating site from pieces.json")
    print()

    # Load data
    pieces = load_pieces()
    print(f"  Loaded {len(pieces)} pieces from pieces.json")

    with open(INDEX_TEMPLATE, "r") as f:
        index_template = f.read()
    with open(PLAYER_TEMPLATE, "r") as f:
        player_template = f.read()

    # Build
    index_html = build_index(pieces, index_template)
    player_html = build_player(pieces, player_template)

    if check_mode:
        # Compare against existing files
        ok = True
        for path, content, name in [
            (INDEX_OUTPUT, index_html, "index.html"),
            (PLAYER_OUTPUT, player_html, "player.html"),
        ]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    existing = f.read()
                if existing == content:
                    print(f"  ✓ {name} matches")
                else:
                    print(f"  ✗ {name} differs")
                    ok = False
            else:
                print(f"  ✗ {name} does not exist")
                ok = False

        if ok:
            print("\n  All files match — build is up to date.")
        else:
            print("\n  Files differ — run without --check to regenerate.")
            sys.exit(1)
    else:
        # Write output
        with open(INDEX_OUTPUT, "w") as f:
            f.write(index_html)
        print(f"  Written: {INDEX_OUTPUT}")

        with open(PLAYER_OUTPUT, "w") as f:
            f.write(player_html)
        print(f"  Written: {PLAYER_OUTPUT}")

        print(f"\n  Done — {len(pieces)} pieces rendered to index.html + player.html")


if __name__ == "__main__":
    main()
