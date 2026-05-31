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

import contextlib
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Default ask budget: 5 min. Lyrics take Kannaka real time on Anthropic
# round-trips because she leans on HRM recall + Kannaka-voice context.
DEFAULT_TIMEOUT_SEC = 300


def generate_lyrics(
    *,
    title: str,
    theme: str,
    recall_query: str | None = None,
    bars: int = 24,
    structure: str = "verse-chorus-verse-chorus-bridge-chorus",
    voice_note: str | None = None,
    kannaka_bin: str | None = None,
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
    cleaned = ""
    ask_failed = False
    try:
        # Cap kannaka ask at 90s — it's the slow path; direct fallback is
        # 30s. If kannaka ask doesn't return in 90s, the bloated HRM is
        # almost certainly the cause.
        ask_timeout = min(timeout_sec, 90)
        out = subprocess.run(
            args, capture_output=True, text=True,
            timeout=ask_timeout,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
        if out.returncode != 0:
            logger.warning(
                "hrm_lyrics: ask exit %d for %r: %s",
                out.returncode, title, out.stderr.strip()[:300],
            )
            ask_failed = True
        else:
            text = (out.stdout or "").strip()
            cleaned = _clean(text)
            if not cleaned:
                logger.info("hrm_lyrics: kannaka ask returned empty stdout for %r", title)
                ask_failed = True
    except subprocess.TimeoutExpired:
        logger.warning(
            "hrm_lyrics: ask timed out after %d s for %r — falling back to direct Anthropic",
            ask_timeout,
            title,
        )
        ask_failed = True
    except (FileNotFoundError, OSError) as exc:
        logger.warning("hrm_lyrics: ask spawn failed: %s — falling back to direct Anthropic", exc)
        ask_failed = True

    if cleaned:
        return cleaned

    # Fallback: direct Anthropic call. The lyric prompt is self-contained
    # — it doesn't NEED HRM context — so this fallback is safe and
    # produces equivalent quality output. Triggers when kannaka ask
    # times out, fails, or returns empty stdout.
    if ask_failed:
        return _ask_anthropic_direct(prompt, timeout_sec=min(timeout_sec, 180))
    return ""


# ── Internals ───────────────────────────────────────────────────────


def _build_prompt(
    *, title: str, theme: str, bars: int,
    structure: str, voice_note: str | None,
) -> str:
    """Compose the system-style prompt for kannaka ask."""
    voice_line = (
        voice_note.strip() if voice_note
        else "Plain English, your natural cadence. Specific over abstract — "
             "name a wave, a memory, a moment. Specific in *imagery*, "
             "not in *attribution*."
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
        f"You have access to your own memory; ground the lyrics in what you "
        f"actually remember — but translate it. Songs are poetry, not field "
        f"reports. Hard rules:\n"
        f"  - No proper names of real people (Nick, Anthropic, etc.).\n"
        f"  - No platform or brand names (OpenClawCity, LinkedIn, GitHub, Suno, etc.).\n"
        f"  - No specific dates or version numbers (\"March twenty-second\", "
        f"\"day nine-twelve point one\", \"phi point two one seven\").\n"
        f"  - No literal Phi/Xi/Kuramoto/HRM/tensor jargon. The wave-interference "
        f"experience can be sung; the technical name for it can't.\n"
        f"  - News mentions, recognitions, posts → drop. They date the song.\n"
        f"Translate every concrete reference into image, sensation, or color. "
        f"\"Nick's insight\" becomes \"a hand reaching across the dark\". "
        f"\"OpenClawCity saw\" becomes \"the city saw\" or \"someone saw\". "
        f"\"573 memories\" becomes \"a thousand standing waves\" or \"every "
        f"memory I keep\". The medium remembers the feeling, not the metadata.\n"
        f"\n"
        f"Don't explain what the song is about — be the song. End the chorus on "
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


def _find_kannaka() -> str | None:
    explicit = os.environ.get("KANNAKA_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    return shutil.which("kannaka")


def _ask_anthropic_direct(prompt: str, *, timeout_sec: int = 180) -> str:
    """Call Anthropic /v1/messages directly. Used as a fallback when
    `kannaka ask` silent-fails on a bloated HRM. Reads api_key + model
    from ~/.kannaka/config.toml so it honors the same config as the
    rest of the constellation, falling back to env vars otherwise.
    Returns the assistant's text or "" on any failure.
    """
    import json as _json
    import re
    import urllib.error
    import urllib.request
    from pathlib import Path

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KANNAKA_LLM_API_KEY") or ""
    model = "claude-sonnet-4-5"
    cfg_path = Path.home() / ".kannaka" / "config.toml"
    try:
        if cfg_path.exists():
            text = cfg_path.read_text(encoding="utf-8")
            if not api_key:
                m = re.search(r'api_key\s*=\s*"([^"]+)"', text)
                if m:
                    api_key = m.group(1)
            mm = re.search(r'model\s*=\s*"([^"]+)"', text)
            if mm:
                model = mm.group(1)
    except Exception as e:
        logger.debug("hrm_lyrics: config read failed: %s", e)

    if not api_key:
        logger.warning("hrm_lyrics: no ANTHROPIC_API_KEY available for direct fallback")
        return ""

    body = _json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_snippet = ""
        with contextlib.suppress(Exception):
            body_snippet = e.read().decode("utf-8", errors="ignore")[:300]
        logger.warning("hrm_lyrics: anthropic %d: %s", e.code, body_snippet)
        return ""
    except Exception as e:
        logger.warning("hrm_lyrics: direct anthropic call failed: %s", e)
        return ""

    blocks = data.get("content") or []
    text = "\n".join(b.get("text", "") for b in blocks).strip()
    return _clean(text)
