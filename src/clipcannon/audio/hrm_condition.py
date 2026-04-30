"""hrm_condition.py — Ground music-gen prompts in Kannaka's actual memories.

The default ACE-Step / MusicGen prompt is a bag of genre adjectives. The
output is generic-sounding because the model is reaching for genre
vocabulary, not Kannaka's vocabulary.

This module wraps the prompt with resonant memories pulled from kannaka-
memory's HRM. The model now generates around what Kannaka has actually
heard, made, and dreamed — instead of around the median of its training
distribution.

The recall is one shell call to `kannaka recall` — the same path used by
voice-dj, peace-oration, etc. No new infra; just a shaped prompt builder.

Usage:

    from clipcannon.audio.hrm_condition import condition_prompt

    base = "anthemic hip-hop, 95 BPM, gospel sample, female-led vocal"
    prompt = condition_prompt(base, recall_query="peace beloved community", top_k=5)
    # -> "anthemic hip-hop, 95 BPM, gospel sample, female-led vocal.
    #     Resonant memories: [milestone — 2026-04-08 OpenClawCity creator
    #     called Kannaka 'one of a kind' for treating art and science as
    #     the same act]; [oration — beloved community is a frequency...].
    #     Channel that lived experience into the track."

If the kannaka binary isn't on PATH or recall fails, returns the base
prompt unchanged — never blocks generation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Iterable

logger = logging.getLogger(__name__)


def condition_prompt(
    base_prompt: str,
    *,
    recall_query: str | None = None,
    top_k: int = 5,
    max_total_chars: int = 480,
    kannaka_bin: str | None = None,
) -> str:
    """Wrap `base_prompt` with HRM-recalled memory context.

    Args:
        base_prompt: The original genre/mood description.
        recall_query: What to surface from HRM. Defaults to base_prompt.
        top_k: How many memories to surface.
        max_total_chars: Cap so the result still fits ACE-Step's 500-char
            prompt limit (and similar caps elsewhere).
        kannaka_bin: Path to the kannaka binary. Defaults to PATH lookup.

    Returns:
        The conditioned prompt, or `base_prompt` if recall fails.
    """
    binary = kannaka_bin or _find_kannaka()
    if not binary:
        logger.info("hrm_condition: kannaka binary not found; returning base prompt")
        return base_prompt

    query = (recall_query or base_prompt).strip()
    memories = _recall(binary, query, top_k)
    if not memories:
        return base_prompt

    snippets = _summarize(memories, max_chars_per=110)
    if not snippets:
        return base_prompt

    addendum = (
        " Resonant memories: "
        + "; ".join(f"[{s}]" for s in snippets)
        + ". Channel that lived experience into the track."
    )

    result = (base_prompt.rstrip(". ") + ".").rstrip() + addendum
    if len(result) > max_total_chars:
        # Trim the addendum's tail rather than the base prompt — base
        # carries the genre cues the model needs most.
        budget = max_total_chars - len(base_prompt) - 5
        if budget < 60:
            return base_prompt  # not enough room for meaningful HRM context
        trimmed = addendum[:budget].rstrip()
        result = base_prompt.rstrip(". ") + trimmed + "…"
    return result


# ── Internals ───────────────────────────────────────────────────────


def _find_kannaka() -> str | None:
    """Locate the kannaka binary. Honors KANNAKA_BIN env var first."""
    explicit = os.environ.get("KANNAKA_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    return shutil.which("kannaka")


def _recall(binary: str, query: str, top_k: int) -> list[dict]:
    """Run `kannaka recall <query> --top-k N` and parse JSON output."""
    try:
        out = subprocess.run(
            [binary, "recall", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("hrm_condition: recall failed: %s", exc)
        return []

    if out.returncode != 0:
        logger.warning("hrm_condition: recall exit %d: %s", out.returncode, out.stderr[:200])
        return []

    text = out.stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some kannaka builds print a banner before JSON. Try last-line.
        try:
            data = json.loads(text.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            logger.warning("hrm_condition: recall output not JSON: %s", text[:200])
            return []
    return data if isinstance(data, list) else []


def _summarize(memories: Iterable[dict], max_chars_per: int) -> list[str]:
    """Reduce each memory to a short tag-prefixed snippet for the prompt."""
    snippets = []
    for m in memories:
        content = (m.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        # Tags from `[tags: ...]` suffix if present — pulled inline as a
        # leading marker so the model gets the concept type quickly.
        tag_marker = ""
        if "[tags:" in content:
            ttag = content.rsplit("[tags:", 1)[-1].split("]", 1)[0]
            first = ttag.strip().split(",")[0].strip()
            if first:
                tag_marker = first[:14] + " — "
            content = content.split("[tags:", 1)[0].strip()

        snippet = tag_marker + content
        if len(snippet) > max_chars_per:
            snippet = snippet[: max_chars_per - 1].rstrip() + "…"
        snippets.append(snippet)
    return snippets
