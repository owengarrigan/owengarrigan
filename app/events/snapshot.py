from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.detection.base_detector import DetectionResult


def save_snapshot(
    *,
    frame: np.ndarray,
    detections: list[DetectionResult],
    output_dir: Path,
    camera_name: str,
) -> Path:
    """Draw bounding boxes and save an event snapshot image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_camera_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in camera_name
    )
    snapshot_path = output_dir / f"{timestamp}_{safe_camera_name}.jpg"

    if not cv2.imwrite(str(snapshot_path), annotated):
        raise RuntimeError(f"Failed to write snapshot: {snapshot_path}")

    return snapshot_path
