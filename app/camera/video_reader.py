import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Read frames from either a local video file or an RTSP/HTTP stream."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None

    @property
    def is_stream(self) -> bool:
        """Return True when the source looks like a network stream."""

        scheme = urlparse(self.source).scheme.lower()
        return scheme in {"rtsp", "rtmp", "http", "https"}

    def open(self) -> None:
        """Open the configured video source and fail fast on bad input."""

        capture_source = self.source
        if not self.is_stream:
            video_path = Path(self.source).expanduser()
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")
            capture_source = str(video_path)

        logger.info("Opening video source: %s", self.source)
        self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield frame number and frame until the source ends or disconnects."""

        if self.capture is None:
            self.open()

        assert self.capture is not None
        frame_number = 0

        while True:
            ok, frame = self.capture.read()
            if not ok:
                logger.warning("Video source ended or frame read failed")
                break

            frame_number += 1
            yield frame_number, frame

    def release(self) -> None:
        """Release the underlying OpenCV capture handle."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
