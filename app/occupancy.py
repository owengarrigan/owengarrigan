"""Occupancy counting and dwell time tracking for Vigil.

Tracks people entering/exiting zones and monitors how long objects remain.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class OccupancyZone:
    """A zone that tracks current occupancy via in/out counting."""
    id: str
    name: str
    current_count: int = 0
    total_in: int = 0
    total_out: int = 0
    peak_count: int = 0
    peak_time: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    def person_entered(self) -> None:
        self.current_count += 1
        self.total_in += 1
        if self.current_count > self.peak_count:
            self.peak_count = self.current_count
            self.peak_time = datetime.now(UTC).isoformat()
        self._log("in")

    def person_exited(self) -> None:
        self.current_count = max(0, self.current_count - 1)
        self.total_out += 1
        self._log("out")

    def _log(self, direction: str) -> None:
        self.history.append({
            "time": datetime.now(UTC).isoformat(),
            "direction": direction,
            "count": self.current_count,
        })
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "current_count": self.current_count,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "peak_count": self.peak_count,
            "peak_time": self.peak_time,
            "recent_history": self.history[-20:],
        }


@dataclass
class DwellTracker:
    """Tracks how long objects remain in a zone."""
    zone_id: str
    zone_name: str
    alert_threshold_seconds: float = 300  # 5 minutes default
    active_objects: dict[int, float] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    average_dwell: float = 0.0
    total_tracked: int = 0
    completed_dwells: list[float] = field(default_factory=list)

    def object_entered(self, object_id: int) -> None:
        self.active_objects[object_id] = time.time()

    def object_exited(self, object_id: int) -> float:
        if object_id not in self.active_objects:
            return 0.0
        entered_at = self.active_objects.pop(object_id)
        dwell_time = time.time() - entered_at
        self.completed_dwells.append(dwell_time)
        self.total_tracked += 1
        if self.completed_dwells:
            self.average_dwell = sum(self.completed_dwells[-100:]) / len(self.completed_dwells[-100:])
        if len(self.completed_dwells) > 500:
            self.completed_dwells = self.completed_dwells[-500:]
        return dwell_time

    def check_alerts(self) -> list[dict[str, Any]]:
        now = time.time()
        new_alerts = []
        for obj_id, entered_at in list(self.active_objects.items()):
            duration = now - entered_at
            if duration > self.alert_threshold_seconds:
                already_alerted = any(
                    a["object_id"] == obj_id for a in self.alerts[-50:]
                )
                if not already_alerted:
                    alert = {
                        "object_id": obj_id,
                        "zone": self.zone_name,
                        "duration_seconds": round(duration),
                        "threshold": self.alert_threshold_seconds,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "type": "dwell_exceeded",
                    }
                    self.alerts.append(alert)
                    new_alerts.append(alert)
        return new_alerts

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "alert_threshold_seconds": self.alert_threshold_seconds,
            "active_objects": len(self.active_objects),
            "average_dwell_seconds": round(self.average_dwell, 1),
            "total_tracked": self.total_tracked,
            "recent_alerts": self.alerts[-10:],
        }


class OccupancyManager:
    """Manages multiple occupancy zones and dwell trackers."""

    def __init__(self) -> None:
        self.zones: dict[str, OccupancyZone] = {}
        self.dwell_trackers: dict[str, DwellTracker] = {}

    def add_zone(self, zone_id: str, name: str, dwell_threshold: float = 300) -> None:
        self.zones[zone_id] = OccupancyZone(id=zone_id, name=name)
        self.dwell_trackers[zone_id] = DwellTracker(
            zone_id=zone_id, zone_name=name, alert_threshold_seconds=dwell_threshold
        )

    def record_entry(self, zone_id: str, object_id: int = 0) -> None:
        if zone_id in self.zones:
            self.zones[zone_id].person_entered()
        if zone_id in self.dwell_trackers:
            self.dwell_trackers[zone_id].object_entered(object_id)

    def record_exit(self, zone_id: str, object_id: int = 0) -> float:
        if zone_id in self.zones:
            self.zones[zone_id].person_exited()
        dwell = 0.0
        if zone_id in self.dwell_trackers:
            dwell = self.dwell_trackers[zone_id].object_exited(object_id)
        return dwell

    def get_all_occupancy(self) -> dict[str, Any]:
        total_current = sum(z.current_count for z in self.zones.values())
        return {
            "total_occupancy": total_current,
            "zones": [z.to_dict() for z in self.zones.values()],
            "dwell_trackers": [d.to_dict() for d in self.dwell_trackers.values()],
        }

    def get_zone_occupancy(self, zone_id: str) -> dict[str, Any] | None:
        zone = self.zones.get(zone_id)
        if zone:
            return zone.to_dict()
        return None

    def check_all_dwell_alerts(self) -> list[dict[str, Any]]:
        all_alerts = []
        for tracker in self.dwell_trackers.values():
            alerts = tracker.check_alerts()
            all_alerts.extend(alerts)
        return all_alerts
