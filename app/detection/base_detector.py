from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Normalized model output used by rules and plugins."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class BaseDetector(ABC):
    """Interface for detector implementations such as YOLO or future plugins."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return detections for a single video frame."""
