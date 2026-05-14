import logging
from collections.abc import Iterable

import numpy as np
from ultralytics import YOLO

from app.detection.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """Ultralytics YOLO detector wrapper with normalized outputs."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        target_labels: Iterable[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_labels = set(target_labels or [])
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run inference on one frame and return matching detections."""

        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)
        if not results:
            return []

        detections: list[DetectionResult] = []
        names = results[0].names

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id))
            confidence = float(box.conf[0])

            if self.target_labels and label not in self.target_labels:
                continue
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                DetectionResult(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        logger.debug("YOLO produced %d matching detections", len(detections))
        return detections
