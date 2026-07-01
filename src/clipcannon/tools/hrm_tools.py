"""HRM (kannaka-memory) MCP tools for Kannaka Cannon.

Exposes two tools:

* ``clipcannon_hrm_recall`` — cross-project semantic recall over every
  analyzed video, backed by ``kannaka recall``.
* ``clipcannon_hrm_ingest`` — (re)store a project's stem outputs
  (transcripts, scene descriptions, music features) into the HRM. The
  pipeline does this automatically on finalize; this tool is for manual
  backfill of previously analyzed projects.

Both degrade gracefully when the kannaka binary is absent, returning
``hrm_available: false`` rather than erroring.
"""

from __future__ import annotations

import asyncio
import logging

from mcp.types import Tool

from clipcannon import hrm_bridge
from clipcannon.tools.understanding import _db_path, _error, _validate_project

logger = logging.getLogger(__name__)


HRM_TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="clipcannon_hrm_recall",
        description=(
            "Recall video moments from the Holographic Resonance Memory "
            "across ALL analyzed projects by natural-language query. Backed "
            "by the kannaka binary; returns hrm_available=false if it is not "
            "installed (set KANNAKA_BIN)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language recall query"},
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="clipcannon_hrm_ingest",
        description=(
            "Store a project's stem outputs (transcripts, scene descriptions, "
            "music features) into the HRM as recallable memories. Runs "
            "automatically after ingest; use this to backfill an already-"
            "analyzed project. Returns hrm_available=false if the kannaka "
            "binary is not installed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project identifier"},
            },
            "required": ["project_id"],
        },
    ),
]


async def clipcannon_hrm_recall(query: str, top_k: int = 5) -> dict[str, object]:
    """Recall video moments from the HRM across all projects.

    Args:
        query: Natural-language recall query.
        top_k: Maximum number of results.

    Returns:
        Dict with results and an hrm_available flag.
    """
    if not query or not query.strip():
        return _error("INVALID_PARAMETER", "query must be a non-empty string")

    if not hrm_bridge.hrm_available():
        return {
            "query": query,
            "hrm_available": False,
            "results": [],
            "count": 0,
            "message": "kannaka binary not found; set KANNAKA_BIN to enable recall.",
        }

    results = await asyncio.to_thread(hrm_bridge.recall, query, top_k=top_k)
    return {
        "query": query,
        "hrm_available": True,
        "results": results,
        "count": len(results),
    }


async def clipcannon_hrm_ingest(project_id: str) -> dict[str, object]:
    """Store a project's stem outputs into the HRM.

    Args:
        project_id: Project identifier.

    Returns:
        Dict with per-type counts and an hrm_available flag.
    """
    err = _validate_project(project_id, required_status="ready")
    if err is not None:
        return err

    if not hrm_bridge.hrm_available():
        return {
            "project_id": project_id,
            "hrm_available": False,
            "stored": 0,
            "message": "kannaka binary not found; set KANNAKA_BIN to enable ingest.",
        }

    summary = await asyncio.to_thread(
        hrm_bridge.ingest_project, project_id, _db_path(project_id)
    )
    return {"hrm_available": True, **summary}


async def dispatch_hrm_tool(
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Dispatch an HRM tool call by name.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool result dictionary.
    """
    if name == "clipcannon_hrm_recall":
        return await clipcannon_hrm_recall(
            str(arguments["query"]),
            int(arguments.get("top_k", 5)),  # type: ignore[arg-type]
        )
    if name == "clipcannon_hrm_ingest":
        return await clipcannon_hrm_ingest(str(arguments["project_id"]))

    return _error("INTERNAL_ERROR", f"Unknown tool: {name}")
