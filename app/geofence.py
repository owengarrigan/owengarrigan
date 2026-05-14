"""Geofencing for Vigil — GPS-based automatic arm/disarm.

When the owner's phone leaves the property → auto-arm
When they return → auto-disarm
Uses a simple radius check against configured home coordinates.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GEOFENCE_FILE = Path("./data/geofence.json")


def load_geofence_config() -> dict[str, Any]:
    if GEOFENCE_FILE.exists():
        return json.loads(GEOFENCE_FILE.read_text())
    return {
        "enabled": False,
        "home_lat": 0.0,
        "home_lon": 0.0,
        "radius_meters": 100,
        "auto_arm_on_leave": True,
        "auto_disarm_on_arrive": True,
        "devices": [],
        "history": [],
    }


def save_geofence_config(config: dict[str, Any]) -> None:
    GEOFENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GEOFENCE_FILE.write_text(json.dumps(config, indent=2))


def configure_geofence(
    home_lat: float,
    home_lon: float,
    radius_meters: int = 100,
    auto_arm: bool = True,
    auto_disarm: bool = True,
) -> dict[str, Any]:
    """Configure the geofence parameters."""
    config = load_geofence_config()
    config["enabled"] = True
    config["home_lat"] = home_lat
    config["home_lon"] = home_lon
    config["radius_meters"] = radius_meters
    config["auto_arm_on_leave"] = auto_arm
    config["auto_disarm_on_arrive"] = auto_disarm
    save_geofence_config(config)
    return config


def check_location(lat: float, lon: float, device_id: str = "phone") -> dict[str, Any]:
    """Check a device location against the geofence.

    Returns whether the device is inside/outside and any triggered actions.
    """
    config = load_geofence_config()
    if not config["enabled"]:
        return {"status": "disabled"}

    distance = _haversine(lat, lon, config["home_lat"], config["home_lon"])
    is_inside = distance <= config["radius_meters"]

    previous_inside = _get_previous_state(device_id, config)

    action = None
    if previous_inside and not is_inside and config["auto_arm_on_leave"]:
        action = "arm"
    elif not previous_inside and is_inside and config["auto_disarm_on_arrive"]:
        action = "disarm"

    entry = {
        "device_id": device_id,
        "lat": lat,
        "lon": lon,
        "distance_meters": round(distance),
        "inside": is_inside,
        "action": action,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    config["history"].append(entry)
    if len(config["history"]) > 500:
        config["history"] = config["history"][-500:]
    save_geofence_config(config)

    return entry


def _get_previous_state(device_id: str, config: dict[str, Any]) -> bool:
    """Get the previous inside/outside state for a device."""
    for entry in reversed(config.get("history", [])):
        if entry.get("device_id") == device_id:
            return entry.get("inside", True)
    return True


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c
