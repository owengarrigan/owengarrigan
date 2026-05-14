"""Simple object tracking and heatmap accumulation for Vigil."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    id: int
    label: str
    positions: list[tuple[int, int]] = field(default_factory=list)
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def speed_pixels_per_frame(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        p1 = self.positions[-2]
        p2 = self.positions[-1]
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @property
    def total_distance(self) -> float:
        dist = 0.0
        for i in range(1, len(self.positions)):
            p1, p2 = self.positions[i-1], self.positions[i]
            dist += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return dist


class SimpleTracker:
    """IOU-based tracker for connecting detections across frames."""

    def __init__(self, max_disappeared: int = 30) -> None:
        self.max_disappeared = max_disappeared
        self._next_id = 0
        self._objects: dict[int, TrackedObject] = {}
        self._disappeared: dict[int, int] = {}

    def update(self, detections: list[dict[str, Any]], frame_number: int) -> list[TrackedObject]:
        """Update tracked objects with new detections."""
        if not detections:
            for obj_id in list(self._disappeared.keys()):
                self._disappeared[obj_id] += 1
                if self._disappeared[obj_id] > self.max_disappeared:
                    del self._objects[obj_id]
                    del self._disappeared[obj_id]
            return list(self._objects.values())

        new_centers = []
        new_bboxes = []
        new_labels = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            new_centers.append((cx, cy))
            new_bboxes.append(tuple(bbox))
            new_labels.append(det["label"])

        if not self._objects:
            for i, (center, bbox, label) in enumerate(zip(new_centers, new_bboxes, new_labels)):
                self._register(center, bbox, label, frame_number)
        else:
            obj_ids = list(self._objects.keys())
            obj_centers = [self._objects[oid].center for oid in obj_ids]

            distances = np.zeros((len(obj_centers), len(new_centers)))
            for i, oc in enumerate(obj_centers):
                for j, nc in enumerate(new_centers):
                    distances[i, j] = math.sqrt((oc[0]-nc[0])**2 + (oc[1]-nc[1])**2)

            used_rows = set()
            used_cols = set()

            flat_indices = np.argsort(distances, axis=None)
            for flat_idx in flat_indices:
                row = int(flat_idx // len(new_centers))
                col = int(flat_idx % len(new_centers))
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > 200:
                    break
                obj_id = obj_ids[row]
                self._objects[obj_id].positions.append(new_centers[col])
                self._objects[obj_id].bbox = new_bboxes[col]
                self._objects[obj_id].last_seen_frame = frame_number
                self._disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in range(len(obj_ids)):
                if row not in used_rows:
                    obj_id = obj_ids[row]
                    self._disappeared[obj_id] += 1
                    if self._disappeared[obj_id] > self.max_disappeared:
                        del self._objects[obj_id]
                        del self._disappeared[obj_id]

            for col in range(len(new_centers)):
                if col not in used_cols:
                    self._register(new_centers[col], new_bboxes[col], new_labels[col], frame_number)

        return list(self._objects.values())

    def _register(self, center: tuple[int, int], bbox: tuple, label: str, frame: int) -> None:
        self._objects[self._next_id] = TrackedObject(
            id=self._next_id,
            label=label,
            positions=[center],
            first_seen_frame=frame,
            last_seen_frame=frame,
            bbox=bbox,
        )
        self._disappeared[self._next_id] = 0
        self._next_id += 1


class HeatmapAccumulator:
    """Accumulates detection positions into a heatmap grid."""

    def __init__(self, width: int = 64, height: int = 48) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.total_detections = 0

    def add_detection(self, x: int, y: int, frame_width: int, frame_height: int) -> None:
        """Add a detection at pixel coordinates, mapped to the grid."""
        gx = int(x / frame_width * self.width)
        gy = int(y / frame_height * self.height)
        gx = max(0, min(gx, self.width - 1))
        gy = max(0, min(gy, self.height - 1))

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    self.grid[ny, nx] += max(0, 1.0 - dist * 0.3)

        self.total_detections += 1

    def get_normalized(self) -> list[list[float]]:
        """Return the heatmap as a normalized 2D array (0-1)."""
        if self.grid.max() == 0:
            return self.grid.tolist()
        normalized = self.grid / self.grid.max()
        return normalized.tolist()

    def get_hotspots(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the top N hotspot positions."""
        flat = self.grid.flatten()
        indices = flat.argsort()[-top_n:][::-1]
        hotspots = []
        for idx in indices:
            if flat[idx] == 0:
                break
            y, x = divmod(int(idx), self.width)
            hotspots.append({
                "x": x / self.width,
                "y": y / self.height,
                "intensity": float(flat[idx] / self.grid.max()) if self.grid.max() > 0 else 0,
            })
        return hotspots

    def reset(self) -> None:
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_detections = 0
