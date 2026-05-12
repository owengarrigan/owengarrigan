from app.detection.base_detector import DetectionResult
from app.rules.base_rule import BaseRule, RuleEvent


class PersonPresentRule(BaseRule):
    """Create an event whenever a person is detected above the threshold."""

    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(self, detections: list[DetectionResult]) -> RuleEvent | None:
        people = [
            detection
            for detection in detections
            if detection.label == "person"
            and detection.confidence >= self.confidence_threshold
        ]
        if not people:
            return None

        best_confidence = max(detection.confidence for detection in people)
        labels = sorted({detection.label for detection in people})
        return RuleEvent(
            event_type="person_present",
            confidence=best_confidence,
            labels_detected=labels,
            detections=people,
        )
