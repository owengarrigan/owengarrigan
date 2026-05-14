from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.detection.base_detector import DetectionResult


@dataclass(frozen=True)
class RuleEvent:
    """An event candidate emitted by a rule before it is persisted."""

    event_type: str
    confidence: float
    labels_detected: list[str]
    detections: list[DetectionResult]


class BaseRule(ABC):
    """Base interface for detection rules."""

    @abstractmethod
    def evaluate(self, detections: list[DetectionResult]) -> RuleEvent | None:
        """Return an event when the rule condition is met."""
