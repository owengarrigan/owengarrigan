from abc import ABC, abstractmethod

from app.detection.base_detector import BaseDetector
from app.rules.base_rule import BaseRule


class BasePlugin(ABC):
    """Plugin contract for future domain-specific detection packages."""

    name: str

    @abstractmethod
    def build_detector(self) -> BaseDetector:
        """Create the detector used by this plugin."""

    @abstractmethod
    def build_rules(self) -> list[BaseRule]:
        """Create rules that turn plugin detections into events."""
