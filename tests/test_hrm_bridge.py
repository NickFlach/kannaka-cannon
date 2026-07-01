"""Tests for the Cannon -> HRM bridge (clipcannon.hrm_bridge) and MCP tools.

Two flavors of "mocked binary":

* A real fake ``kannaka`` executable (a tiny launcher + Python script) is
  created on disk and pointed to via KANNAKA_BIN. It logs ``remember``
  calls and emits canned JSON for ``recall`` — exercising the full
  subprocess + discovery + parse path. ASCII-only content is used here to
  stay clear of Windows ``.bat`` quoting.
* ``subprocess.run`` is monkeypatched for the detailed command-construction
  and edge-case assertions (importance/tags/modality, timeouts, banner
  parsing) where inspecting the exact argv is clearer than round-tripping.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from clipcannon import hrm_bridge
from clipcannon.pipeline.finalize import _hrm_ingest_best_effort
from clipcannon.tools import hrm_tools

if TYPE_CHECKING:
    from pathlib import Path

# A minimal stand-in for the kannaka binary. `remember` appends its argv to
# FAKE_KANNAKA_LOG; `recall` prints FAKE_KANNAKA_RECALL (or a default list).
_FAKE_SCRIPT = """\
import json, os, sys
argv = sys.argv[1:]
log = os.environ.get("FAKE_KANNAKA_LOG")
if argv[:1] == ["remember"]:
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\\n")
    sys.exit(0)
if argv[:1] == ["recall"]:
    out = os.environ.get("FAKE_KANNAKA_RECALL")
    print(out if out is not None else json.dumps([{"content": "fake memory", "score": 0.9}]))
    sys.exit(0)
sys.exit(2)
"""


@pytest.fixture()
def fake_kannaka(tmp_path: Path) -> SimpleNamespace:
    """Create a real fake kannaka executable and its call log."""
    script = tmp_path / "fake_kannaka.py"
    script.write_text(_FAKE_SCRIPT, encoding="utf-8")
    log = tmp_path / "calls.log"

    if os.name == "nt":
        launcher = tmp_path / "kannaka.bat"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
        )
    else:
        launcher = tmp_path / "kannaka"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(0o755)

    return SimpleNamespace(bin=str(launcher), log=log)


def _logged_calls(log: Path) -> list[list[str]]:
    """Parse the fake binary's recorded remember argv lists."""
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


# ── Binary discovery ────────────────────────────────────────────────
class TestFindKannaka:
    def test_env_var_honored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = tmp_path / "kannaka-bin"
        binary.write_text("x", encoding="utf-8")
        monkeypatch.setenv("KANNAKA_BIN", str(binary))
        assert hrm_bridge.find_kannaka() == str(binary)
        assert hrm_bridge.hrm_available() is True

    def test_env_var_missing_file_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KANNAKA_BIN", str(tmp_path / "does-not-exist"))
        monkeypatch.setattr(hrm_bridge.shutil, "which", lambda _: None)
        assert hrm_bridge.find_kannaka() is None
        assert hrm_bridge.hrm_available() is False

    def test_path_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KANNAKA_BIN", raising=False)
        monkeypatch.setattr(hrm_bridge.shutil, "which", lambda _: "/usr/bin/kannaka")
        assert hrm_bridge.find_kannaka() == "/usr/bin/kannaka"

    def test_explicit_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = tmp_path / "explicit"
        binary.write_text("x", encoding="utf-8")
        monkeypatch.delenv("KANNAKA_BIN", raising=False)
        assert hrm_bridge.find_kannaka(str(binary)) == str(binary)


# ── remember ────────────────────────────────────────────────────────
class TestRemember:
    def test_no_binary_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KANNAKA_BIN", raising=False)
        monkeypatch.setattr(hrm_bridge.shutil, "which", lambda _: None)
        assert hrm_bridge.remember("anything") is False

    def test_empty_content_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        called = False

        def _fail(*_a: object, **_k: object) -> object:
            nonlocal called
            called = True
            raise AssertionError("should not run")

        monkeypatch.setattr(hrm_bridge.subprocess, "run", _fail)
        assert hrm_bridge.remember("   ") is False
        assert called is False

    def test_command_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(hrm_bridge.subprocess, "run", _fake_run)

        ok = hrm_bridge.remember(
            "a moment", importance=0.75, tags=["cannon", "transcript"], modality="semantic"
        )
        assert ok is True
        cmd = captured["cmd"]
        assert cmd[:3] == ["/fake/kannaka", "remember", "a moment"]
        assert "--importance" in cmd and cmd[cmd.index("--importance") + 1] == "0.75"
        assert "--tags" in cmd and cmd[cmd.index("--tags") + 1] == "cannon,transcript"
        assert "--modality" in cmd and cmd[cmd.index("--modality") + 1] == "semantic"
        assert "--category" in cmd and cmd[cmd.index("--category") + 1] == "cannon"

    def test_nonzero_exit_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        )
        assert hrm_bridge.remember("x") is False

    def test_timeout_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")

        def _raise(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired(cmd="kannaka", timeout=1)

        monkeypatch.setattr(hrm_bridge.subprocess, "run", _raise)
        assert hrm_bridge.remember("x") is False

    def test_real_binary_roundtrip(
        self, fake_kannaka: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KANNAKA_BIN", fake_kannaka.bin)
        monkeypatch.setenv("FAKE_KANNAKA_LOG", str(fake_kannaka.log))
        ok = hrm_bridge.remember(
            "integration smoke memory", importance=0.6, tags=["cannon", "transcript"]
        )
        assert ok is True
        calls = _logged_calls(fake_kannaka.log)
        assert len(calls) == 1
        assert calls[0][:3] == ["remember", "integration smoke memory", "--importance"]


# ── recall ──────────────────────────────────────────────────────────
class TestRecall:
    def test_no_binary_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KANNAKA_BIN", raising=False)
        monkeypatch.setattr(hrm_bridge.shutil, "which", lambda _: None)
        assert hrm_bridge.recall("q") == []

    def test_parses_json_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps([{"content": "m1"}, {"content": "m2"}])
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, payload, ""),
        )
        results = hrm_bridge.recall("query", top_k=2)
        assert len(results) == 2
        assert results[0]["content"] == "m1"

    def test_banner_then_json_last_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = "kannaka v0.6\nloading HRM...\n" + json.dumps([{"content": "m"}])
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, payload, ""),
        )
        assert hrm_bridge.recall("q") == [{"content": "m"}]

    def test_non_json_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "not json at all", ""),
        )
        assert hrm_bridge.recall("q") == []

    def test_real_binary_recall(
        self, fake_kannaka: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KANNAKA_BIN", fake_kannaka.bin)
        monkeypatch.setenv(
            "FAKE_KANNAKA_RECALL", json.dumps([{"content": "hit", "score": 0.8}])
        )
        results = hrm_bridge.recall("anything", top_k=3)
        assert results == [{"content": "hit", "score": 0.8}]


# ── ingest_project ──────────────────────────────────────────────────
def _seed_db(db_path: Path, project_id: str = "proj_hrm") -> None:
    """Create an analysis DB with transcript, scene, and music rows."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE project (project_id TEXT PRIMARY KEY, name TEXT, duration_ms INTEGER);
        CREATE TABLE transcript_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
            start_ms INTEGER, end_ms INTEGER, text TEXT);
        CREATE TABLE scenes (
            scene_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
            start_ms INTEGER, end_ms INTEGER, shot_type TEXT,
            dominant_colors TEXT, face_detected BOOLEAN, quality_classification TEXT);
        CREATE TABLE beats (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
            has_music BOOLEAN, tempo_bpm REAL, beat_count INTEGER);
        CREATE TABLE music_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
            start_ms INTEGER, end_ms INTEGER, type TEXT);
        """
    )
    conn.execute("INSERT INTO project VALUES (?, ?, ?)", (project_id, "Demo Video", 6000))
    conn.executemany(
        "INSERT INTO transcript_segments (project_id, start_ms, end_ms, text) VALUES (?,?,?,?)",
        [
            (project_id, 0, 2000, "hello there"),
            (project_id, 2000, 4000, "welcome to the show"),
            (project_id, 4000, 6000, "   "),  # blank -> skipped by query
        ],
    )
    conn.executemany(
        "INSERT INTO scenes (project_id, start_ms, end_ms, shot_type, dominant_colors, "
        "face_detected, quality_classification) VALUES (?,?,?,?,?,?,?)",
        [
            (project_id, 0, 3000, "medium", "#112233,#445566", 1, "good"),
            (project_id, 3000, 6000, "wide", "#778899", 0, "fair"),
        ],
    )
    conn.execute(
        "INSERT INTO beats (project_id, has_music, tempo_bpm, beat_count) VALUES (?,?,?,?)",
        (project_id, 1, 120.0, 240),
    )
    conn.execute(
        "INSERT INTO music_sections (project_id, start_ms, end_ms, type) VALUES (?,?,?,?)",
        (project_id, 0, 6000, "intro"),
    )
    conn.commit()
    conn.close()


class TestIngestProject:
    def test_no_binary_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        monkeypatch.delenv("KANNAKA_BIN", raising=False)
        monkeypatch.setattr(hrm_bridge.shutil, "which", lambda _: None)
        result = hrm_bridge.ingest_project("proj_hrm", db)
        assert result["available"] is False
        assert result["stored"] == 0

    def test_stores_summary_per_stem_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        recorded: list[list[str]] = []

        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            recorded.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(hrm_bridge.subprocess, "run", _fake_run)

        result = hrm_bridge.ingest_project("proj_hrm", db)

        # One concise summary each: transcript + scenes + music = 3.
        assert result["available"] is True
        assert result["transcript"] == 1
        assert result["scenes"] == 1
        assert result["music"] == 1
        assert result["stored"] == 3
        assert result["failed"] == 0
        assert len(recorded) == 3

        # Modality routing: transcript semantic, scenes visual, music audio.
        modalities = [cmd[cmd.index("--modality") + 1] for cmd in recorded]
        assert sorted(modalities) == ["audio", "semantic", "visual"]

        # Every memory is tagged with the project id and the cannon namespace.
        for cmd in recorded:
            tags = cmd[cmd.index("--tags") + 1]
            assert "proj_hrm" in tags
            assert "cannon" in tags

        # Transcript is a single summary that folds in multiple segments.
        contents = [cmd[2] for cmd in recorded]
        tx = next(c for c in contents if "transcript summary" in c)
        assert "hello there" in tx and "welcome to the show" in tx
        assert any("scenes (2 over" in c and "medium" in c for c in contents)
        assert any("music:" in c and "120 BPM" in c for c in contents)

    def test_transcript_summary_is_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        conn = sqlite3.connect(str(db))
        conn.execute("DELETE FROM transcript_segments")
        conn.execute(
            "INSERT INTO transcript_segments (project_id, start_ms, end_ms, text) "
            "VALUES (?,?,?,?)",
            ("proj_hrm", 0, 1000, "word " * 800),  # ~4000 chars of transcript
        )
        conn.commit()
        conn.close()

        recorded: list[list[str]] = []
        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: recorded.append(cmd)  # type: ignore[func-returns-value]
            or subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        hrm_bridge.ingest_project("proj_hrm", db, max_chars=200)

        tx = next(c[2] for c in recorded if "transcript summary" in c[2])
        assert tx.endswith("…")  # ellipsis -> truncated
        assert len(tx) < 400  # prefix + ~200-char body, nowhere near 4000

    def test_no_music_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        # Flip has_music off.
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE beats SET has_music = 0")
        conn.commit()
        conn.close()

        monkeypatch.setattr(hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        result = hrm_bridge.ingest_project("proj_hrm", db)
        assert result["music"] == 0
        assert result["stored"] == 2  # transcript + scenes

    def test_real_binary_ingest(
        self, tmp_path: Path, fake_kannaka: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        monkeypatch.setenv("KANNAKA_BIN", fake_kannaka.bin)
        monkeypatch.setenv("FAKE_KANNAKA_LOG", str(fake_kannaka.log))
        result = hrm_bridge.ingest_project("proj_hrm", db)
        assert result["stored"] == 3
        assert len(_logged_calls(fake_kannaka.log)) == 3


class TestIngestEnabled:
    def test_default_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLIPCANNON_HRM_INGEST", raising=False)
        assert hrm_bridge.ingest_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "no"])
    def test_explicit_off(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLIPCANNON_HRM_INGEST", val)
        assert hrm_bridge.ingest_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes"])
    def test_explicit_on(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLIPCANNON_HRM_INGEST", val)
        assert hrm_bridge.ingest_enabled() is True


class TestFinalizeHook:
    """The pipeline hook is fire-and-forget: it must never raise."""

    @pytest.mark.asyncio
    async def test_never_raises_when_ingest_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLIPCANNON_HRM_INGEST", "1")
        monkeypatch.setattr(hrm_bridge, "hrm_available", lambda *_a, **_k: True)

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("kannaka exploded")

        monkeypatch.setattr(hrm_bridge, "ingest_project", _boom)
        # Must complete without propagating the error.
        await _hrm_ingest_best_effort("proj", tmp_path / "analysis.db")

    @pytest.mark.asyncio
    async def test_skips_when_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLIPCANNON_HRM_INGEST", "0")
        calls = {"n": 0}

        def _count(*_a: object, **_k: object) -> dict[str, object]:
            calls["n"] += 1
            return {}

        monkeypatch.setattr(hrm_bridge, "hrm_available", lambda *_a, **_k: True)
        monkeypatch.setattr(hrm_bridge, "ingest_project", _count)
        await _hrm_ingest_best_effort("proj", tmp_path / "analysis.db")
        assert calls["n"] == 0


# ── MCP tools ───────────────────────────────────────────────────────
class TestHrmTools:
    @pytest.mark.asyncio
    async def test_recall_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_tools.hrm_bridge, "hrm_available", lambda *_a, **_k: False)
        out = await hrm_tools.clipcannon_hrm_recall("find pricing talk")
        assert out["hrm_available"] is False
        assert out["results"] == []
        assert out["count"] == 0

    @pytest.mark.asyncio
    async def test_recall_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hrm_tools.hrm_bridge, "hrm_available", lambda *_a, **_k: True)
        monkeypatch.setattr(
            hrm_tools.hrm_bridge, "recall", lambda *_a, **_k: [{"content": "hit"}]
        )
        out = await hrm_tools.clipcannon_hrm_recall("q", top_k=3)
        assert out["hrm_available"] is True
        assert out["count"] == 1
        assert out["results"][0]["content"] == "hit"

    @pytest.mark.asyncio
    async def test_recall_rejects_empty_query(self) -> None:
        out = await hrm_tools.clipcannon_hrm_recall("   ")
        assert "error" in out

    @pytest.mark.asyncio
    async def test_ingest_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "analysis.db"
        _seed_db(db)
        monkeypatch.setattr(hrm_tools, "_validate_project", lambda *_a, **_k: None)
        monkeypatch.setattr(hrm_tools, "_db_path", lambda _pid: db)
        monkeypatch.setattr(hrm_tools.hrm_bridge, "hrm_available", lambda *_a, **_k: True)
        monkeypatch.setattr(hrm_tools.hrm_bridge, "find_kannaka", lambda *_a, **_k: "/fake/kannaka")
        monkeypatch.setattr(
            hrm_tools.hrm_bridge.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        out = await hrm_tools.clipcannon_hrm_ingest("proj_hrm")
        assert out["hrm_available"] is True
        assert out["stored"] == 3

    @pytest.mark.asyncio
    async def test_ingest_tool_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hrm_tools, "_validate_project", lambda *_a, **_k: None)
        monkeypatch.setattr(hrm_tools.hrm_bridge, "hrm_available", lambda *_a, **_k: False)
        out = await hrm_tools.clipcannon_hrm_ingest("proj_hrm")
        assert out["hrm_available"] is False
        assert out["stored"] == 0
