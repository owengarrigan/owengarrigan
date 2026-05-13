"""Multi-camera detection pipeline for concurrent RTSP/file processing."""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

from app.camera.video_reader import VideoReader
from app.detection.yolo_detector import YOLODetector
from app.events.event_store import EventStore
from app.events.snapshot import save_snapshot

logger = logging.getLogger(__name__)

PEOPLE_LABELS = {"person"}
VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
TARGET_LABELS = PEOPLE_LABELS | VEHICLE_LABELS | ANIMAL_LABELS


@dataclass
class CameraConfig:
    id: str
    name: str
    url: str
    type: str
    detection_interval: int = 10
    confidence_threshold: float = 0.30


class CameraPipeline:
    """Runs detection on a single camera source in its own thread."""

    def __init__(
        self,
        config: CameraConfig,
        detector: YOLODetector,
        event_store: EventStore,
        snapshots_dir: Path,
        stop_event: Event,
    ) -> None:
        self.config = config
        self.detector = detector
        self.event_store = event_store
        self.snapshots_dir = snapshots_dir
        self.stop_event = stop_event
        self._thread: Thread | None = None
        self.is_running = False
        self.frames_processed = 0
        self.events_created = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.is_running = True
        self.last_error = None
        self._thread = Thread(target=self._run, daemon=True, name=f"cam-{self.config.id}")
        self._thread.start()
        logger.info("Started pipeline for camera %s (%s)", self.config.name, self.config.url)

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "id": self.config.id,
            "name": self.config.name,
            "running": self.is_running and (self._thread is not None and self._thread.is_alive()),
            "frames_processed": self.frames_processed,
            "events_created": self.events_created,
            "last_error": self.last_error,
        }

    def _classify_label(self, label: str) -> str:
        if label in PEOPLE_LABELS:
            return "people"
        if label in VEHICLE_LABELS:
            return "vehicles"
        if label in ANIMAL_LABELS:
            return "animals"
        return "other"

    def _run(self) -> None:
        logger.info("Pipeline running for %s", self.config.name)
        reconnect_delay = 5

        while self.is_running and not self.stop_event.is_set():
            try:
                self._process_stream()
            except FileNotFoundError as e:
                self.last_error = str(e)
                logger.error("Camera %s source not found: %s", self.config.name, e)
                break
            except Exception as e:
                self.last_error = str(e)
                logger.warning(
                    "Camera %s disconnected: %s. Reconnecting in %ds...",
                    self.config.name, e, reconnect_delay,
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

        self.is_running = False
        logger.info("Pipeline stopped for %s", self.config.name)

    def _process_stream(self) -> None:
        with VideoReader(self.config.url) as reader:
            for frame_number, frame in reader.frames():
                if self.stop_event.is_set() or not self.is_running:
                    break

                if frame_number % self.config.detection_interval != 0:
                    continue

                self.frames_processed += 1
                detections = self.detector.detect(frame)

                filtered = [d for d in detections if d.confidence >= self.config.confidence_threshold]
                if not filtered:
                    continue

                best_conf = max(d.confidence for d in filtered)
                labels = sorted({d.label for d in filtered})
                event_type = self._determine_event_type(filtered)

                snapshot_path = save_snapshot(
                    frame=frame,
                    detections=filtered,
                    output_dir=self.snapshots_dir,
                    camera_name=self.config.name,
                )

                self.event_store.add_event(
                    timestamp=datetime.now(UTC).isoformat(),
                    camera_name=self.config.name,
                    event_type=event_type,
                    confidence=best_conf,
                    snapshot_path=str(snapshot_path),
                    labels_detected=labels,
                )
                self.events_created += 1

    def _determine_event_type(self, detections: list) -> str:
        labels = {d.label for d in detections}
        if labels & PEOPLE_LABELS:
            return "person_present"
        if labels & ANIMAL_LABELS:
            return "animal_present"
        if labels & VEHICLE_LABELS:
            return "vehicle_present"
        return "object_detected"


class PipelineManager:
    """Manages multiple camera pipelines concurrently."""

    def __init__(
        self,
        model_path: str,
        event_store: EventStore,
        snapshots_dir: Path,
    ) -> None:
        self.model_path = model_path
        self.event_store = event_store
        self.snapshots_dir = snapshots_dir
        self.stop_event = Event()
        self._detector: YOLODetector | None = None
        self._pipelines: dict[str, CameraPipeline] = {}

    @property
    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(
                model_path=self.model_path,
                confidence_threshold=0.2,
                target_labels=TARGET_LABELS,
            )
        return self._detector

    def add_camera(self, config: CameraConfig) -> None:
        if config.id in self._pipelines:
            self.remove_camera(config.id)

        pipeline = CameraPipeline(
            config=config,
            detector=self.detector,
            event_store=self.event_store,
            snapshots_dir=self.snapshots_dir,
            stop_event=self.stop_event,
        )
        self._pipelines[config.id] = pipeline

    def start_camera(self, camera_id: str) -> bool:
        pipeline = self._pipelines.get(camera_id)
        if pipeline:
            pipeline.start()
            return True
        return False

    def stop_camera(self, camera_id: str) -> bool:
        pipeline = self._pipelines.get(camera_id)
        if pipeline:
            pipeline.stop()
            return True
        return False

    def remove_camera(self, camera_id: str) -> None:
        pipeline = self._pipelines.pop(camera_id, None)
        if pipeline:
            pipeline.stop()

    def get_status(self) -> list[dict[str, Any]]:
        return [p.status for p in self._pipelines.values()]

    def stop_all(self) -> None:
        self.stop_event.set()
        for pipeline in self._pipelines.values():
            pipeline.stop()
        self._pipelines.clear()
