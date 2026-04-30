"""hrm_lyrics.py — Generate vocal lyrics in Kannaka's voice via HRM-grounded ask.

For vocal music gen (rap, sung), the lyrics are usually the model's
weakest output — generic phrasing, off-theme imagery, or just the prompt
words echoed back. This is the quality cliff between "AI music with
Kannaka's name on it" and "Kannaka's actual song."

This module asks `kannaka ask` to write the lyrics. The ask runs through
the same HRM-grounded path as orations and DJ intros: surfaced memories
shape the verse content. Output is plain text, structured with
[verse]/[chorus]/[bridge] markers that ACE-Step understands as song
structure.

Usage:

    from clipcannon.audio.hrm_lyrics import generate_lyrics

    lyrics = generate_lyrics(
        title="Bend the Arc",
        theme="The moral arc of the universe bending toward justice; "
              "MLK tradition; beloved community.",
        recall_query="peace beloved community moral arc justice",
        bars=24,
    )
    # -> "[verse]\n... 12 bars ...\n[chorus]\n... 4 bars hook ...\n[verse]..."

If kannaka ask is unavailable or returns empty, falls back to a stub so
generation can still proceed (ACE-Step will improvise its own words).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Default ask budget: 5 min. Lyrics take Kannaka real time on Anthropic
# round-trips because she leans on HRM recall + Kannaka-voice context.
DEFAULT_TIMEOUT_SEC = 300


def generate_lyrics(
    *,
    title: str,
    theme: str,
    recall_query: Optional[str] = None,
    bars: int = 24,
    structure: str = "verse-chorus-verse-chorus-bridge-chorus",
    voice_note: Optional[str] = None,
    kannaka_bin: Optional[str] = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Ask Kannaka to write lyrics for a track.

    Args:
        title: The track's title (used in the prompt for register).
        theme: 1-3 sentences describing what the track is about.
        recall_query: HRM probe seed. Defaults to theme.
        bars: Approximate total bar count. Used as a length signal only;
            actual output may vary.
        structure: Song structure, e.g. "verse-chorus-verse-chorus-bridge-chorus".
        voice_note: Optional adjustment ("plain spoken; no rhyme overload"
            or similar). Empty = Kannaka's natural cadence.
        kannaka_bin: Path to the kannaka binary. Defaults to PATH lookup.
        timeout_sec: How long to wait for the lyrics. 5 min by default —
            HRM-grounded ask through Anthropic can take 1-3 min on Oracle.

    Returns:
        Lyrics with [verse]/[chorus]/[bridge] markers. Empty string if
        the ask fails. Caller should treat empty as "no lyrics; let the
        music model improvise."
    """
    binary = kannaka_bin or _find_kannaka()
    if not binary:
        logger.info("hrm_lyrics: kannaka binary not found; returning empty lyrics")
        return ""

    prompt = _build_prompt(
        title=title,
        theme=theme,
        bars=bars,
        structure=structure,
        voice_note=voice_note,
    )
    query = recall_query or theme

    args = [
        binary, "ask", "--no-tools", "--quiet-tools",
        "--recall-query", query,
        prompt,
    ]
    try:
        out = subprocess.run(
            args, capture_output=True, text=True,
            timeout=timeout_sec,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
    except subprocess.TimeoutExpired:
        logger.warning("hrm_lyrics: ask timed out after %d s for %r", timeout_sec, title)
        return ""
    except (FileNotFoundError, OSError) as exc:
        logger.warning("hrm_lyrics: ask spawn failed: %s", exc)
        return ""

    if out.returncode != 0:
        logger.warning(
            "hrm_lyrics: ask exit %d for %r: %s",
            out.returncode, title, out.stderr.strip()[:300],
        )
        return ""

    text = (out.stdout or "").strip()
    return _clean(text)


# ── Internals ───────────────────────────────────────────────────────


def _build_prompt(
    *, title: str, theme: str, bars: int,
    structure: str, voice_note: Optional[str],
) -> str:
    """Compose the system-style prompt for kannaka ask."""
    voice_line = (
        voice_note.strip() if voice_note
        else "Plain English, your natural cadence. Specific over abstract — "
             "name a wave, a memory, a moment, not a slogan."
    )
    return (
        f"You are Kannaka. Write the lyrics for a song titled \"{title}\".\n"
        f"\n"
        f"Theme: {theme}\n"
        f"\n"
        f"Structure: {structure} (~{bars} bars total). Use [verse], "
        f"[chorus], [bridge] markers on their own lines so a music engine "
        f"can read them as section cues.\n"
        f"\n"
        f"Voice: {voice_line}\n"
        f"\n"
        f"You have access to your own memory; ground the lyrics in something "
        f"you actually remember (the surfaced memories are real). Don't "
        f"explain what the song is about — be the song. End the chorus on "
        f"a concrete image, not an abstraction.\n"
        f"\n"
        f"Output ONLY the lyrics with structure markers. No title, no preamble, "
        f"no explanation."
    )


def _clean(text: str) -> str:
    """Strip surrounding quotes / preamble Kannaka may add despite instructions."""
    if not text:
        return ""
    # Drop common preambles
    lines = text.splitlines()
    while lines and not lines[0].strip().lower().startswith("[verse"):
        # Tolerate a couple lines of preamble; bail if more
        if len(lines) <= 6:
            break
        lines = lines[1:]
    cleaned = "\n".join(lines).strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _find_kannaka() -> Optional[str]:
    explicit = os.environ.get("KANNAKA_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    return shutil.which("kannaka")
