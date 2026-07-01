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
from typing import TYPE_CHECKING

from clipcannon.db.connection import get_connection
from clipcannon.db.queries import fetch_all, fetch_one

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_PER_TYPE = 500

# Importance seeds per stem type (see docs/hrm-integration.md).
_IMPORTANCE = {
    "transcript": 0.6,
    "scene": 0.5,
    "music": 0.5,
}


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


def _project_title(db_path: Path, project_id: str) -> str:
    """Best-effort project display name."""
    conn = get_connection(db_path, enable_vec=False, dict_rows=True)
    try:
        row = fetch_one(
            conn, "SELECT name FROM project WHERE project_id = ?", (project_id,)
        )
    finally:
        conn.close()
    if row and row.get("name"):
        return str(row["name"])
    return project_id


def _store_all(
    memories: Iterable[tuple[str, float, list[str], str]],
    *,
    kannaka_bin: str | None,
) -> tuple[int, int]:
    """Store a batch of (content, importance, tags, modality) tuples.

    Returns:
        (stored, failed) counts.
    """
    stored = 0
    failed = 0
    for content, importance, tags, modality in memories:
        ok = remember(
            content,
            importance=importance,
            tags=tags,
            modality=modality,
            kannaka_bin=kannaka_bin,
        )
        if ok:
            stored += 1
        else:
            failed += 1
    return stored, failed


def ingest_project(
    project_id: str,
    db_path: Path,
    *,
    kannaka_bin: str | None = None,
    max_per_type: int = _DEFAULT_MAX_PER_TYPE,
) -> dict[str, object]:
    """Store a project's stem outputs into the HRM.

    Reads transcript segments, scene descriptions, and music/beat features
    from the project's analysis DB and stores each as an HRM memory tagged
    with the project id. No-op (``available: False``) if the kannaka binary
    cannot be located, so callers never need to guard.

    Args:
        project_id: Project identifier.
        db_path: Path to the project's ``analysis.db``.
        kannaka_bin: Explicit binary path override.
        max_per_type: Cap on memories stored per stem type (avoids flooding
            the HRM on long videos).

    Returns:
        Dict with per-type counts and totals.
    """
    binary = find_kannaka(kannaka_bin)
    if not binary:
        logger.info("HRM ingest skipped: kannaka binary not found")
        return {"available": False, "stored": 0, "failed": 0}

    title = _project_title(db_path, project_id)

    conn = get_connection(db_path, enable_vec=False, dict_rows=True)
    try:
        transcripts = fetch_all(
            conn,
            "SELECT start_ms, end_ms, text FROM transcript_segments "
            "WHERE project_id = ? AND text IS NOT NULL AND trim(text) != '' "
            "ORDER BY start_ms LIMIT ?",
            (project_id, max_per_type),
        )
        scenes = fetch_all(
            conn,
            "SELECT start_ms, end_ms, shot_type, dominant_colors, "
            "face_detected, quality_classification FROM scenes "
            "WHERE project_id = ? ORDER BY start_ms LIMIT ?",
            (project_id, max_per_type),
        )
        beats = fetch_one(
            conn,
            "SELECT has_music, tempo_bpm, beat_count FROM beats "
            "WHERE project_id = ? LIMIT 1",
            (project_id,),
        )
        sections = fetch_all(
            conn,
            "SELECT type FROM music_sections WHERE project_id = ? "
            "AND type IS NOT NULL ORDER BY start_ms LIMIT ?",
            (project_id, max_per_type),
        )
    finally:
        conn.close()

    counts = {"transcripts": 0, "scenes": 0, "music": 0}
    total_stored = 0
    total_failed = 0

    # Transcripts → semantic memories
    tx_batch: list[tuple[str, float, list[str], str]] = []
    for seg in transcripts:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        content = (
            f"[cannon:{project_id}] {title} transcript "
            f"({_fmt_ts(seg.get('start_ms'))}-{_fmt_ts(seg.get('end_ms'))}): {text}"
        )
        tx_batch.append(
            (content, _IMPORTANCE["transcript"], ["cannon", "transcript", project_id], "semantic")
        )
    stored, failed = _store_all(tx_batch, kannaka_bin=binary)
    counts["transcripts"] = stored
    total_stored += stored
    total_failed += failed

    # Scene descriptions → visual memories
    sc_batch: list[tuple[str, float, list[str], str]] = []
    for scene in scenes:
        content = (
            f"[cannon:{project_id}] {title} scene "
            f"({_fmt_ts(scene.get('start_ms'))}-{_fmt_ts(scene.get('end_ms'))}): "
            f"{_scene_description(scene)}"
        )
        sc_batch.append(
            (content, _IMPORTANCE["scene"], ["cannon", "scene", project_id], "visual")
        )
    stored, failed = _store_all(sc_batch, kannaka_bin=binary)
    counts["scenes"] = stored
    total_stored += stored
    total_failed += failed

    # Music / beat features → one audio summary memory
    if beats and beats.get("has_music"):
        tempo = beats.get("tempo_bpm")
        tempo_str = f"{float(tempo):.0f} BPM" if tempo is not None else "unknown tempo"
        section_types = [str(s.get("type")) for s in sections if s.get("type")]
        section_str = (
            f", sections: {', '.join(section_types)}" if section_types else ""
        )
        content = (
            f"[cannon:{project_id}] {title} music: {tempo_str}, "
            f"{beats.get('beat_count') or 0} beats{section_str}"
        )
        stored, failed = _store_all(
            [(content, _IMPORTANCE["music"], ["cannon", "music", project_id], "audio")],
            kannaka_bin=binary,
        )
        counts["music"] = stored
        total_stored += stored
        total_failed += failed

    result = {
        "available": True,
        "project_id": project_id,
        "stored": total_stored,
        "failed": total_failed,
        **counts,
    }
    logger.info("HRM ingest for %s: %s", project_id, result)
    return result
