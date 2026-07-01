"""Bridge Kannaka Cannon's video analysis into the HRM (kannaka-memory).

After an ingest completes, stem outputs — transcript segments, scene
descriptions, and music/beat features — are stored as HRM memories via
``kannaka remember``, so any constellation member can ``kannaka recall``
across every analyzed video. Recall is exposed as an MCP tool.

Everything here is guarded behind the presence of the kannaka binary
(``KANNAKA_BIN`` env var, else ``kannaka`` on PATH). If it isn't found,
ingest is a no-op and recall returns an empty list — the video pipeline
never depends on the HRM being available. This mirrors the shell-out
pattern already used by ``audio/hrm_condition.py`` and ``hrm_lyrics.py``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from typing import TYPE_CHECKING

from clipcannon.db.connection import get_connection
from clipcannon.db.queries import fetch_all, fetch_one

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_MEMORY_CHARS = 600  # cap on composed memory text — a summary, not full text
_MAX_SCENE_SAMPLES = 3  # sample scene descriptions folded into the scene summary
_MUSIC_IMPORTANCE = 0.5


# ── Binary discovery ────────────────────────────────────────────────
def find_kannaka(kannaka_bin: str | None = None) -> str | None:
    """Locate the kannaka binary. Honors KANNAKA_BIN, then PATH.

    Args:
        kannaka_bin: Explicit path override; used if it points at a file.

    Returns:
        Path to the binary, or None if not found.
    """
    explicit = kannaka_bin or os.environ.get("KANNAKA_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    return shutil.which("kannaka")


def hrm_available(kannaka_bin: str | None = None) -> bool:
    """Return True if the kannaka binary can be located."""
    return find_kannaka(kannaka_bin) is not None


def ingest_enabled() -> bool:
    """Whether automatic pipeline HRM ingest is enabled.

    Controlled by ``CLIPCANNON_HRM_INGEST``; defaults ON (so ingest runs
    whenever the kannaka binary is present). Set it to ``0``/``false``/``off``
    to disable the automatic post-finalize ingest. The explicit
    ``clipcannon_hrm_ingest`` MCP tool ignores this gate.
    """
    val = os.environ.get("CLIPCANNON_HRM_INGEST")
    if val is None:
        return True
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Memory write / read ─────────────────────────────────────────────
def remember(
    content: str,
    *,
    importance: float = 0.5,
    tags: Sequence[str] | None = None,
    category: str = "cannon",
    modality: str = "mixed",
    kannaka_bin: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> bool:
    """Store one memory via ``kannaka remember``.

    Args:
        content: The memory text (positional arg to the binary).
        importance: Seed amplitude / salience (0..1).
        tags: Comma-joined into ``--tags``.
        category: ``--category`` value (defaults to ``cannon``).
        modality: ``--modality`` value (semantic|audio|visual|network|mixed).
        kannaka_bin: Explicit binary path override.
        timeout: Subprocess timeout in seconds.

    Returns:
        True if the memory was stored, False on any failure or missing binary.
    """
    binary = find_kannaka(kannaka_bin)
    if not binary:
        return False

    text = (content or "").strip()
    if not text:
        return False

    cmd: list[str] = [
        binary,
        "remember",
        text,
        "--importance",
        f"{importance:.2f}",
        "--category",
        category,
        "--modality",
        modality,
    ]
    if tags:
        cmd += ["--tags", ",".join(t for t in tags if t)]

    try:
        out = subprocess.run(  # noqa: S603 - binary path is operator-controlled
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("hrm remember failed: %s", exc)
        return False

    if out.returncode != 0:
        logger.warning("hrm remember exit %d: %s", out.returncode, out.stderr[:200])
        return False
    return True


def recall(
    query: str,
    *,
    top_k: int = 5,
    kannaka_bin: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> list[dict]:
    """Run ``kannaka recall <query> --top-k N`` and parse JSON output.

    Args:
        query: Natural-language recall query.
        top_k: Number of results to request.
        kannaka_bin: Explicit binary path override.
        timeout: Subprocess timeout in seconds.

    Returns:
        List of memory dicts, or an empty list on any failure / missing binary.
    """
    binary = find_kannaka(kannaka_bin)
    if not binary:
        return []

    q = (query or "").strip()
    if not q:
        return []

    try:
        out = subprocess.run(  # noqa: S603 - binary path is operator-controlled
            [binary, "recall", q, "--top-k", str(top_k)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "KANNAKA_QUIET": "1"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("hrm recall failed: %s", exc)
        return []

    if out.returncode != 0:
        logger.warning("hrm recall exit %d: %s", out.returncode, out.stderr[:200])
        return []

    text = out.stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some kannaka builds print a banner before JSON. Try last line.
        try:
            data = json.loads(text.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            logger.warning("hrm recall output not JSON: %s", text[:200])
            return []
    return data if isinstance(data, list) else []


# ── Project ingest ──────────────────────────────────────────────────
def _fmt_ts(ms: object) -> str:
    """Format a millisecond value as ``M:SS`` (best-effort)."""
    try:
        total_s = int(ms) // 1000  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"
    return f"{total_s // 60}:{total_s % 60:02d}"


def _scene_description(row: dict[str, object]) -> str:
    """Synthesize a natural-language description from a scenes row."""
    shot = str(row.get("shot_type") or "").strip()
    parts: list[str] = [f"{shot} shot" if shot else "scene"]
    colors = str(row.get("dominant_colors") or "").strip()
    if colors:
        parts.append(f"dominant colors {colors}")
    parts.append("face visible" if row.get("face_detected") else "no face")
    quality = str(row.get("quality_classification") or "").strip()
    if quality:
        parts.append(f"quality {quality}")
    return ", ".join(parts)


def _cap(text: str, limit: int) -> str:
    """Collapse whitespace and truncate ``text`` to ``limit`` chars."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _transcript_importance(word_count: int) -> float:
    """More spoken content -> higher salience (capped)."""
    return round(min(0.85, 0.50 + word_count / 5000.0), 2)


def _scene_importance(scene_count: int) -> float:
    """More distinct scenes -> higher salience (capped)."""
    return round(min(0.70, 0.40 + scene_count / 100.0), 2)


def _summarize_scenes(
    project_id: str,
    title: str,
    duration: str,
    scenes: list[dict[str, object]],
) -> str:
    """Compose one concise memory describing a project's scenes."""
    shots = Counter(str(s.get("shot_type") or "unknown") for s in scenes)
    shot_str = ", ".join(f"{name}x{n}" for name, n in shots.most_common())
    samples = "; ".join(_scene_description(s) for s in scenes[:_MAX_SCENE_SAMPLES])
    body = _cap(f"shot types {shot_str}. samples: {samples}", _MAX_MEMORY_CHARS)
    return f"[cannon:{project_id}] {title} scenes ({len(scenes)} over {duration}): {body}"


def _summarize_music(
    project_id: str,
    title: str,
    beats: dict[str, object],
    sections: list[dict[str, object]],
) -> str:
    """Compose one concise memory describing a project's music features."""
    tempo = beats.get("tempo_bpm")
    tempo_str = f"{float(tempo):.0f} BPM" if tempo is not None else "unknown tempo"
    section_types = [str(s.get("type")) for s in sections if s.get("type")]
    section_str = f", sections: {', '.join(section_types)}" if section_types else ""
    body = _cap(
        f"{tempo_str}, {beats.get('beat_count') or 0} beats{section_str}", _MAX_MEMORY_CHARS
    )
    return f"[cannon:{project_id}] {title} music: {body}"


def ingest_project(
    project_id: str,
    db_path: Path,
    *,
    kannaka_bin: str | None = None,
    max_chars: int = _MAX_MEMORY_CHARS,
) -> dict[str, object]:
    """Store a project's stem outputs into the HRM as concise summaries.

    Composes at most one memory per stem type — a transcript summary
    (capped, not full text), a scene summary, and a music summary — each
    tagged ``cannon,<type>,<project_id>`` with an importance heuristic, then
    stores them via ``kannaka remember``. No-op (``available: False``) if the
    kannaka binary cannot be located, so callers never need to guard.

    Args:
        project_id: Project identifier.
        db_path: Path to the project's ``analysis.db``.
        kannaka_bin: Explicit binary path override.
        max_chars: Cap on the salient-content portion of each memory.

    Returns:
        Dict with per-type flags (0/1) and stored/failed totals.
    """
    binary = find_kannaka(kannaka_bin)
    if not binary:
        logger.debug("HRM ingest skipped for %s: kannaka binary not found", project_id)
        return {
            "available": False,
            "stored": 0,
            "failed": 0,
            "transcript": 0,
            "scenes": 0,
            "music": 0,
        }

    conn = get_connection(db_path, enable_vec=False, dict_rows=True)
    try:
        proj = fetch_one(
            conn,
            "SELECT name, duration_ms FROM project WHERE project_id = ?",
            (project_id,),
        )
        transcripts = fetch_all(
            conn,
            "SELECT text FROM transcript_segments "
            "WHERE project_id = ? AND text IS NOT NULL AND trim(text) != '' "
            "ORDER BY start_ms",
            (project_id,),
        )
        scenes = fetch_all(
            conn,
            "SELECT start_ms, end_ms, shot_type, dominant_colors, "
            "face_detected, quality_classification FROM scenes "
            "WHERE project_id = ? ORDER BY start_ms",
            (project_id,),
        )
        beats = fetch_one(
            conn,
            "SELECT has_music, tempo_bpm, beat_count FROM beats WHERE project_id = ? LIMIT 1",
            (project_id,),
        )
        sections = fetch_all(
            conn,
            "SELECT type FROM music_sections WHERE project_id = ? "
            "AND type IS NOT NULL ORDER BY start_ms",
            (project_id,),
        )
    finally:
        conn.close()

    title = str(proj["name"]) if proj and proj.get("name") else project_id
    duration = _fmt_ts(proj.get("duration_ms")) if proj else "?"

    # (content, importance, tags, modality, kind) — at most one per stem type.
    memories: list[tuple[str, float, list[str], str, str]] = []

    texts = [t for t in (str(s.get("text") or "").strip() for s in transcripts) if t]
    if texts:
        word_count = sum(len(t.split()) for t in texts)
        body = _cap(" ".join(texts), max_chars)
        content = (
            f"[cannon:{project_id}] {title} transcript summary "
            f"({duration}, {word_count} words): {body}"
        )
        memories.append(
            (
                content,
                _transcript_importance(word_count),
                ["cannon", "transcript", project_id],
                "semantic",
                "transcript",
            )
        )

    if scenes:
        memories.append(
            (
                _summarize_scenes(project_id, title, duration, scenes),
                _scene_importance(len(scenes)),
                ["cannon", "scene", project_id],
                "visual",
                "scenes",
            )
        )

    if beats and beats.get("has_music"):
        memories.append(
            (
                _summarize_music(project_id, title, beats, sections),
                _MUSIC_IMPORTANCE,
                ["cannon", "music", project_id],
                "audio",
                "music",
            )
        )

    counts = {"transcript": 0, "scenes": 0, "music": 0}
    stored = 0
    failed = 0
    for content, importance, tags, modality, kind in memories:
        if remember(
            content, importance=importance, tags=tags, modality=modality, kannaka_bin=binary
        ):
            stored += 1
            counts[kind] = 1
        else:
            failed += 1

    result = {
        "available": True,
        "project_id": project_id,
        "stored": stored,
        "failed": failed,
        **counts,
    }
    logger.info("HRM ingest for %s: %s", project_id, result)
    return result
