from app.detection.base_detector import BaseDetector
from app.detection.yolo_detector import YOLODetector
from app.plugins.base_plugin import BasePlugin
from app.rules.base_rule import BaseRule


class PPEPlugin(BasePlugin):
    """Placeholder plugin showing how a PPE model can be swapped in later."""

    name = "ppe"

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def build_detector(self) -> BaseDetector:
        return YOLODetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            target_labels={"hardhat", "hi-vis", "person"},
        )

    def build_rules(self) -> list[BaseRule]:
        # Add PPE-specific rules here once a hardhat/hi-vis model is available.
        return []
