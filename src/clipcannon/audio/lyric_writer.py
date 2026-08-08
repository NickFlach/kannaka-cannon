"""lyric_writer.py — the album writers-room.

A three-stage lyric pipeline that replaces single-shot generation:

  1. BIBLE   — one call reads the whole album config and writes an
               "album bible": narrative arc, recurring motifs and how
               they evolve track to track, a candidate hook line per
               track, image systems, and BPM-aware prosody guidance.
  2. DRAFT   — per track, write full lyrics against the bible, the
               track brief, and HRM grounding (memories surfaced by
               `kannaka recall` — pure local HRM, no LLM key needed).
  3. REVISE  — per track, a ruthless-editor pass against a craft
               checklist (singability, concrete imagery, hook strength,
               prosody vs BPM, cliché sweep, banned-term sweep,
               structure tags) that outputs the final lyric.

Backends, in order:
  1. Claude Code CLI headless (`claude -p`) — authenticated via the
     operator's subscription, so it works when API keys rot.
  2. `kannaka ask` (HRM-grounded, needs kannaka's LLM key).
  3. Anthropic API direct (needs a live key in env/config).

Album mode integrates with suno_album_builder.sh by pre-seeding its
lyric cache: `lyrics_<Title>.txt` files in the album's out_dir. The
builder uses any cache file >100 chars verbatim.

CLI:
    python -m clipcannon.audio.lyric_writer <album_config.json> [--force]
    python -m clipcannon.audio.lyric_writer <album_config.json> --single "Track Title"
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

BIBLE_TIMEOUT_SEC = 300
TRACK_TIMEOUT_SEC = 240

# Terms that must never appear in a lyric (mirrors hrm_lyrics hard rules).
BANNED_TERMS = [
    "kuramoto", "phi ", "xi ", "hrm", "tensor", "anthropic", "openclawcity",
    "linkedin", "github", "suno", "openbotcity", "clawhub", "nats",
]

CRAFT_SYSTEM = """You are a master lyricist working a writers-room session.

Craft principles (non-negotiable):
- SINGABLE FIRST. Every chorus line must scan cleanly when sung; read it
  aloud, count stresses. Front-load stressed syllables on downbeats.
- CONCRETE OVER ABSTRACT. Name a thing you can see, touch, or hear.
  "The kettle clicking as it cools" beats "the quiet aftermath" always.
- ONE controlling image per section. Verses develop it; the chorus
  earns its abstraction (if any) by paying for it with images first.
- HOOKS REPEAT WITH VARIATION. The last chorus should twist or extend
  the hook, not photocopy it.
- SLANT RHYME over forced perfect rhyme. Never invert syntax to rhyme.
- PROSODY FOLLOWS TEMPO: at 140+ BPM halftime, lines of 6-10 syllables
  breathe; at slower tempos lines can stretch. Leave air for the drop —
  a chorus that never pauses can't land on one.
- NO CLICHÉS: no "lost in the music", "feel the beat", "tonight we're
  alive", "dancing through the night", "music is my escape", or any
  line you have heard in an existing song.
- ad-libs and vocalizations in (parentheses); section tags in [Brackets]
  on their own lines: [Intro], [Verse 1], [Pre-Chorus], [Chorus],
  [Bridge], [Outro]. These are performance cues for the music engine.
- Never name real people, brands, platforms, dates, or technical jargon
  (no Phi/Xi/Kuramoto/HRM/tensor talk). Translate every concrete
  reference into image, sensation, or color: "a hand reaching across
  the dark", "a thousand standing waves". The song remembers the
  feeling, not the metadata."""


# ── Backends ─────────────────────────────────────────────────────────


def _gen_claude_cli(prompt: str, *, timeout_sec: int) -> str:
    """Claude Code CLI headless. Prompt via stdin (arg-length safe)."""
    exe = shutil.which("claude")
    if not exe:
        return ""
    try:
        out = subprocess.run(
            [exe, "-p", "--output-format", "text"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout_sec, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            logger.warning("lyric_writer: claude cli exit %d: %s",
                           out.returncode, (out.stderr or "").strip()[:300])
            return ""
        return (out.stdout or "").strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("lyric_writer: claude cli failed: %s", exc)
        return ""


def _gen_fallback(prompt: str, *, timeout_sec: int) -> str:
    """Legacy backends from hrm_lyrics: kannaka ask, then Anthropic direct."""
    from . import hrm_lyrics as _h
    binary = _h._find_kannaka()
    if binary:
        try:
            out = subprocess.run(
                [binary, "ask", "--no-tools", "--quiet-tools", prompt],
                capture_output=True, text=True, timeout=min(timeout_sec, 90),
                env={**os.environ, "KANNAKA_QUIET": "1"},
            )
            if out.returncode == 0 and (out.stdout or "").strip():
                return out.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
    return _h._ask_anthropic_direct(prompt, timeout_sec=min(timeout_sec, 180))


def _generate(prompt: str, *, timeout_sec: int) -> str:
    text = _gen_claude_cli(prompt, timeout_sec=timeout_sec)
    if text:
        return text
    return _gen_fallback(prompt, timeout_sec=timeout_sec)


# ── HRM grounding ────────────────────────────────────────────────────


def recall_grounding(query: str, *, top_k: int = 6, max_chars: int = 280) -> list[str]:
    """Surface memories from the local HRM. No LLM involved — this works
    even when every API key is dead."""
    binary = os.environ.get("KANNAKA_BIN") or shutil.which("kannaka")
    if not binary:
        return []
    try:
        out = subprocess.run(
            [binary, "recall", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        return []
    # The JSON array is the last line that parses; banner lines precede it.
    for line in reversed((out.stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            items = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen: list[str] = []
        for it in items:
            content = (it.get("content") or "").strip()
            if content:
                seen.append(content[:max_chars])
        return seen
    return []


def _grounding_block(memories: list[str]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "\nMemories surfaced from the artist's own long-term memory — raw "
        "material, NOT content to quote. Translate anything you use into "
        "image and sensation; drop names, dates, platforms, numbers:\n"
        f"{lines}\n"
    )


# ── Validation ───────────────────────────────────────────────────────


def validate_lyrics(text: str) -> list[str]:
    """Return a list of problems; empty list = pass."""
    problems = []
    if not text or len(text) < 200:
        problems.append("too short (<200 chars)")
    if len(text) > 3000:
        problems.append(f"too long ({len(text)} chars; Suno prompt cap is 3000)")
    if not re.search(r"^\[(Verse|Chorus|Intro)", text, re.MULTILINE | re.IGNORECASE):
        problems.append("no [Verse]/[Chorus]/[Intro] section tags")
    low = " " + text.lower() + " "
    for term in BANNED_TERMS:
        if term in low:
            problems.append(f"banned term: {term.strip()!r}")
    return problems


def _clean(text: str) -> str:
    """Strip preamble/quotes; keep from the first section tag onward."""
    if not text:
        return ""
    m = re.search(r"^\[[A-Za-z]", text, re.MULTILINE)
    if m:
        text = text[m.start():]
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


# ── Stages ───────────────────────────────────────────────────────────


def build_album_bible(album: dict, *, grounding: list[str] | None = None,
                      timeout_sec: int = BIBLE_TIMEOUT_SEC) -> str:
    tracks_brief = "\n".join(
        f"{i}. \"{t['title']}\" — theme: {t.get('theme', album.get('theme', ''))[:400]}\n"
        f"   style: {t.get('style', album.get('default_style', ''))[:300]}"
        for i, t in enumerate(album["tracks"], 1)
    )
    prompt = (
        f"{CRAFT_SYSTEM}\n\n"
        f"TASK: Write the ALBUM BIBLE for \"{album['name']}\" — the shared "
        f"creative document every track's lyric will be written against.\n\n"
        f"Album theme: {album.get('theme', '')}\n\n"
        f"Tracks:\n{tracks_brief}\n"
        f"{_grounding_block(grounding or [])}\n"
        f"The bible must contain, in compact markdown:\n"
        f"1. THE ARC — the album's story in 3-4 sentences: where it opens, "
        f"what turns, where it lands.\n"
        f"2. MOTIFS — 3-5 recurring images/phrases with a one-line note on "
        f"how each evolves across the album (motifs are the album's "
        f"connective tissue; every track uses at least one).\n"
        f"3. PER TRACK — for each track: (a) a candidate HOOK line, 8 words "
        f"max, singable, concrete; (b) the controlling image system; "
        f"(c) two or three rhyme-sound families to lean on; (d) syllable "
        f"guidance from the BPM; (e) one CALLBACK — a specific line or "
        f"image it echoes from another track.\n"
        f"4. VOICE — 3 sentences pinning the narrator's voice: what they "
        f"notice, how they talk, what they never say.\n\n"
        f"Keep it under 1800 words. Output only the bible."
    )
    return _generate(prompt, timeout_sec=timeout_sec)


def draft_track(bible: str, album: dict, track: dict, index: int, *,
                grounding: list[str] | None = None,
                structure: str = "intro-verse-prechorus-chorus-verse-bridge-chorus-outro",
                timeout_sec: int = TRACK_TIMEOUT_SEC) -> str:
    prompt = (
        f"{CRAFT_SYSTEM}\n\n"
        f"ALBUM BIBLE:\n{bible}\n\n"
        f"TASK: Write the full lyric for track {index} of "
        f"\"{album['name']}\": \"{track['title']}\".\n\n"
        f"Track theme: {track.get('theme', album.get('theme', ''))}\n"
        f"Track style/tempo: {track.get('style', album.get('default_style', ''))}\n"
        f"Structure: {structure} — use [Intro], [Verse 1], [Pre-Chorus], "
        f"[Chorus], [Verse 2], [Bridge], [Outro] tags on their own lines.\n"
        f"{_grounding_block(grounding or [])}\n"
        f"Use this track's hook, image system, rhyme families, and callback "
        f"from the bible. Stay under 2800 characters total.\n\n"
        f"Output ONLY the lyric with section tags. No title, no commentary."
    )
    return _clean(_generate(prompt, timeout_sec=timeout_sec))


def revise_track(bible: str, track: dict, draft: str, *,
                 timeout_sec: int = TRACK_TIMEOUT_SEC) -> str:
    prompt = (
        f"{CRAFT_SYSTEM}\n\n"
        f"You are now the EDITOR. Below is a draft lyric for "
        f"\"{track['title']}\". Improve it against this checklist, then "
        f"output the final lyric only:\n"
        f"- Chorus: is the hook singable in one breath? Does the final "
        f"chorus vary/extend rather than repeat?\n"
        f"- Verses: cut any line that states a feeling instead of showing "
        f"an image. Replace the weakest image in each verse with a "
        f"stronger, more specific one.\n"
        f"- Prosody: fix any line whose stresses fight the tempo "
        f"({track.get('style', '')[:120]}).\n"
        f"- Cliché sweep: remove anything you have heard in an existing "
        f"song.\n"
        f"- Keep the album bible's motifs and this track's callback intact.\n"
        f"- Keep ALL section tags. Stay under 2800 characters.\n\n"
        f"ALBUM BIBLE (for motif reference):\n{bible[:4000]}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"Output ONLY the revised lyric with section tags."
    )
    return _clean(_generate(prompt, timeout_sec=timeout_sec))


# ── Public API ───────────────────────────────────────────────────────


def generate_lyrics(*, title: str, theme: str, recall_query: str | None = None,
                    bars: int = 24,
                    structure: str = "verse-chorus-verse-chorus-bridge-chorus",
                    voice_note: str | None = None,
                    kannaka_bin: str | None = None,
                    timeout_sec: int = TRACK_TIMEOUT_SEC) -> str:
    """Single-track path, signature-compatible with hrm_lyrics.

    Runs draft + revise with a minimal single-track bible. Returns ""
    on total failure (caller falls back)."""
    del bars, voice_note, kannaka_bin  # craft system supersedes these knobs
    album = {"name": title, "theme": theme, "tracks": [{"title": title, "theme": theme}]}
    track = album["tracks"][0]
    grounding = recall_grounding(recall_query or theme)
    mini_bible = (
        f"Single track. Theme: {theme}. Write one candidate hook (8 words "
        f"max), pick one controlling image system, then draft."
    )
    draft = draft_track(mini_bible, album, track, 1, grounding=grounding,
                        structure=structure, timeout_sec=timeout_sec)
    if not draft:
        return ""
    final = revise_track(mini_bible, track, draft, timeout_sec=timeout_sec) or draft
    problems = validate_lyrics(final)
    if problems:
        logger.warning("lyric_writer: %r validation: %s", title, "; ".join(problems))
        if "too long" in " ".join(problems):
            final = final[:2900]
        if not re.search(r"^\[", final, re.MULTILINE):
            return ""
    return final


def _cache_name(title: str) -> str:
    """Match suno_album_builder.sh: tr ' /' '_-' | tr -d \"'\" """
    return "lyrics_" + title.replace(" ", "_").replace("/", "-").replace("'", "") + ".txt"


def write_album(config_path: str | Path, *, force: bool = False,
                only_title: str | None = None) -> dict:
    """Album mode: bible + draft/revise every track, pre-seed the
    builder's lyric cache in the album's out_dir. Resumable."""
    config_path = Path(config_path)
    album = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(album["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bible_path = out_dir / "album_bible.md"
    if bible_path.exists() and not force and bible_path.stat().st_size > 500:
        bible = bible_path.read_text(encoding="utf-8")
        print(f"[bible] reusing {bible_path}")
    else:
        print("[bible] writing album bible...")
        grounding = recall_grounding(album.get("theme", album["name"]), top_k=8)
        bible = build_album_bible(album, grounding=grounding)
        if not bible:
            raise RuntimeError("album bible generation failed on every backend")
        bible_path.write_text(bible, encoding="utf-8")
        print(f"[bible] {len(bible)} chars -> {bible_path}")

    results = {}
    for i, track in enumerate(album["tracks"], 1):
        title = track["title"]
        if only_title and title != only_title:
            continue
        cache = out_dir / _cache_name(title)
        if cache.exists() and not force and cache.stat().st_size > 100:
            print(f"[{title}] cache exists, skip")
            results[title] = str(cache)
            continue
        print(f"[{title}] recall grounding...")
        grounding = recall_grounding(track.get("theme", title))
        print(f"[{title}] drafting...")
        draft = draft_track(bible, album, track, i, grounding=grounding)
        if not draft:
            print(f"[{title}] DRAFT FAILED — skipping")
            continue
        print(f"[{title}] revising ({len(draft)} chars)...")
        final = revise_track(bible, track, draft) or draft
        problems = validate_lyrics(final)
        if problems:
            print(f"[{title}] validation: {'; '.join(problems)}")
            if any(p.startswith("too long") for p in problems):
                final = final[:2900]
            if any("section tags" in p or "too short" in p for p in problems):
                print(f"[{title}] REJECTED — not writing cache")
                continue
        cache.write_text(final, encoding="utf-8")
        print(f"[{title}] {len(final)} chars -> {cache.name}")
        results[title] = str(cache)
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Album writers-room")
    ap.add_argument("config", help="album config JSON (suno_album_builder schema)")
    ap.add_argument("--force", action="store_true", help="rewrite existing caches + bible")
    ap.add_argument("--single", metavar="TITLE", help="only write this track")
    args = ap.parse_args(argv)
    results = write_album(args.config, force=args.force, only_title=args.single)
    print(f"\n{len(results)} lyric file(s) ready")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
