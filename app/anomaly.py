"""Smart anomaly detection for Vigil.

Instead of alerting on every detection, this module identifies
genuinely unusual activity by learning normal patterns.
"""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any


class AnomalyDetector:
    """Detects unusual activity by comparing against learned baselines."""

    def __init__(self) -> None:
        self.hourly_baselines: defaultdict[int, list[int]] = defaultdict(list)
        self.camera_baselines: defaultdict[str, list[int]] = defaultdict(list)
        self.current_hour_count: int = 0
        self.last_hour: int = -1
        self.anomalies: list[dict[str, Any]] = []

    def record_detection(self, camera_name: str) -> dict[str, Any] | None:
        """Record a detection and check if it's anomalous."""
        now = datetime.now(UTC)
        current_hour = now.hour

        if current_hour != self.last_hour:
            if self.last_hour >= 0:
                self.hourly_baselines[self.last_hour].append(self.current_hour_count)
                if len(self.hourly_baselines[self.last_hour]) > 30:
                    self.hourly_baselines[self.last_hour] = self.hourly_baselines[self.last_hour][-30:]
            self.current_hour_count = 0
            self.last_hour = current_hour

        self.current_hour_count += 1

        anomaly = self._check_anomaly(current_hour, camera_name)
        if anomaly:
            self.anomalies.append(anomaly)
            if len(self.anomalies) > 200:
                self.anomalies = self.anomalies[-200:]
        return anomaly

    def _check_anomaly(self, hour: int, camera: str) -> dict[str, Any] | None:
        """Check if current activity level is anomalous for this hour."""
        baseline_data = self.hourly_baselines.get(hour, [])
        if len(baseline_data) < 3:
            return None

        avg = sum(baseline_data) / len(baseline_data)
        if avg == 0:
            avg = 1

        ratio = self.current_hour_count / avg

        if ratio > 3.0:
            return {
                "type": "high_activity",
                "severity": "warning" if ratio < 5 else "critical",
                "message": f"Activity is {ratio:.1f}x normal for {hour:02d}:00 ({self.current_hour_count} vs avg {avg:.0f})",
                "camera": camera,
                "hour": hour,
                "current": self.current_hour_count,
                "baseline_avg": round(avg, 1),
                "ratio": round(ratio, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        if ratio < 0.2 and self.current_hour_count > 0 and avg > 10:
            return {
                "type": "low_activity",
                "severity": "info",
                "message": f"Unusually quiet: only {self.current_hour_count} detections vs avg {avg:.0f} for {hour:02d}:00",
                "camera": camera,
                "hour": hour,
                "current": self.current_hour_count,
                "baseline_avg": round(avg, 1),
                "ratio": round(ratio, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        return None

    def get_baselines(self) -> dict[str, Any]:
        """Return learned baselines for display."""
        hourly_avg = {}
        for hour, counts in self.hourly_baselines.items():
            if counts:
                hourly_avg[f"{hour:02d}:00"] = round(sum(counts) / len(counts), 1)

        return {
            "hourly_averages": hourly_avg,
            "data_points": sum(len(v) for v in self.hourly_baselines.values()),
            "recent_anomalies": self.anomalies[-10:],
            "current_hour_count": self.current_hour_count,
            "learning_status": "ready" if sum(len(v) for v in self.hourly_baselines.values()) > 24 else "learning",
        }

    def get_activity_heatmap(self) -> list[dict[str, Any]]:
        """Return hourly activity data for a weekly heatmap visualization."""
        heatmap = []
        for hour in range(24):
            data = self.hourly_baselines.get(hour, [])
            avg = sum(data) / len(data) if data else 0
            heatmap.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "average": round(avg, 1),
                "intensity": min(1.0, avg / 50) if avg > 0 else 0,
            })
        return heatmap
