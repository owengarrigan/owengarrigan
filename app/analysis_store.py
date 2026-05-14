"""Persistent storage for video analysis results."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANALYSIS_STORE_FILE = Path("./data/analysis_history.json")


def load_analysis_history() -> list[dict[str, Any]]:
    if ANALYSIS_STORE_FILE.exists():
        return json.loads(ANALYSIS_STORE_FILE.read_text())
    return []


def save_analysis_history(history: list[dict[str, Any]]) -> None:
    ANALYSIS_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_STORE_FILE.write_text(json.dumps(history, indent=2))


def store_analysis_result(
    filename: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Save an analysis result persistently."""
    history = load_analysis_history()

    entry = {
        "id": len(history) + 1,
        "filename": filename,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_frames": result.get("total_frames", 0),
        "frames_analysed": result.get("frames_analysed", 0),
        "object_summary": result.get("object_summary", {}),
        "category_summary": result.get("category_summary", {}),
        "plates_detected": result.get("plates_detected", []),
        "frames_with_detections": [
            f for f in result.get("frames", []) if f.get("detections")
        ],
    }

    history.append(entry)
    if len(history) > 100:
        history = history[-100:]
    save_analysis_history(history)
    return entry


def get_analysis_by_id(analysis_id: int) -> dict[str, Any] | None:
    history = load_analysis_history()
    for entry in history:
        if entry["id"] == analysis_id:
            return entry
    return None
