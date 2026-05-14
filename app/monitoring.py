"""Professional monitoring station for Vigil.

Provides alarm monitoring centre (ARC) functionality:
- Multi-site monitoring with arm/disarm
- Event priority queue with acknowledge/escalate/dismiss
- Operator audit trail
- Client isolation (each customer sees only their site)
"""

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


MONITORING_FILE = Path("./data/monitoring.json")


class ArmState(str, Enum):
    ARMED = "armed"
    DISARMED = "disarmed"
    ARMED_AWAY = "armed_away"
    ARMED_HOME = "armed_home"


class EventPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventAction(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    ESCALATE = "escalate"
    DISMISS = "dismiss"
    INVESTIGATE = "investigate"


def load_monitoring_state() -> dict[str, Any]:
    if MONITORING_FILE.exists():
        return json.loads(MONITORING_FILE.read_text())
    return {
        "sites": {},
        "event_queue": [],
        "audit_log": [],
        "operators": [],
    }


def save_monitoring_state(state: dict[str, Any]) -> None:
    MONITORING_FILE.parent.mkdir(parents=True, exist_ok=True)
    MONITORING_FILE.write_text(json.dumps(state, indent=2))


def set_arm_state(site_id: str, arm_state: str, operator: str = "system") -> dict[str, Any]:
    """Arm or disarm a site."""
    state = load_monitoring_state()
    if site_id not in state["sites"]:
        state["sites"][site_id] = {}

    old_state = state["sites"][site_id].get("arm_state", "disarmed")
    state["sites"][site_id]["arm_state"] = arm_state
    state["sites"][site_id]["arm_changed_at"] = datetime.now(UTC).isoformat()
    state["sites"][site_id]["arm_changed_by"] = operator

    state["audit_log"].append({
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "arm_state_change",
        "site_id": site_id,
        "from": old_state,
        "to": arm_state,
        "operator": operator,
    })

    save_monitoring_state(state)
    return state["sites"][site_id]


def get_arm_state(site_id: str) -> str:
    state = load_monitoring_state()
    return state.get("sites", {}).get(site_id, {}).get("arm_state", "disarmed")


def classify_event_priority(
    event_type: str,
    confidence: float,
    arm_state: str,
    schedule_active: bool = True,
) -> str:
    """Determine event priority based on context."""
    if arm_state in ("armed", "armed_away"):
        if event_type == "person_present" and confidence > 0.5:
            return "critical"
        if event_type == "person_present":
            return "high"
        if event_type == "vehicle_present":
            return "high"
        return "medium"

    if arm_state == "armed_home":
        if event_type == "person_present" and confidence > 0.7:
            return "medium"
        return "low"

    return "info"


def add_event_to_queue(
    site_id: str,
    event_id: int,
    event_type: str,
    camera_name: str,
    confidence: float,
    snapshot_path: str = "",
    priority: str = "medium",
) -> dict[str, Any]:
    """Add a detection event to the monitoring queue."""
    state = load_monitoring_state()

    entry = {
        "id": len(state["event_queue"]) + 1,
        "site_id": site_id,
        "event_id": event_id,
        "event_type": event_type,
        "camera_name": camera_name,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
        "priority": priority,
        "status": "pending",
        "timestamp": datetime.now(UTC).isoformat(),
        "actioned_by": None,
        "actioned_at": None,
        "action": None,
        "notes": "",
    }

    state["event_queue"].append(entry)
    if len(state["event_queue"]) > 5000:
        state["event_queue"] = state["event_queue"][-5000:]

    save_monitoring_state(state)
    return entry


def action_event(
    queue_id: int,
    action: str,
    operator: str = "admin",
    notes: str = "",
) -> dict[str, Any] | None:
    """Take action on a queued event."""
    state = load_monitoring_state()

    for event in state["event_queue"]:
        if event["id"] == queue_id:
            event["status"] = action
            event["actioned_by"] = operator
            event["actioned_at"] = datetime.now(UTC).isoformat()
            event["action"] = action
            event["notes"] = notes

            state["audit_log"].append({
                "timestamp": datetime.now(UTC).isoformat(),
                "action": f"event_{action}",
                "queue_id": queue_id,
                "event_id": event["event_id"],
                "operator": operator,
                "notes": notes,
            })

            if len(state["audit_log"]) > 10000:
                state["audit_log"] = state["audit_log"][-10000:]

            save_monitoring_state(state)
            return event

    return None


def get_pending_events(site_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Get pending events from the monitoring queue."""
    state = load_monitoring_state()
    events = [e for e in state["event_queue"] if e["status"] == "pending"]
    if site_id:
        events = [e for e in events if e["site_id"] == site_id]
    events.sort(key=lambda e: (
        {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(e["priority"], 5),
        e["timestamp"],
    ))
    return events[:limit]


def get_monitoring_dashboard() -> dict[str, Any]:
    """Get the full monitoring station dashboard data."""
    state = load_monitoring_state()
    pending = [e for e in state["event_queue"] if e["status"] == "pending"]
    critical = [e for e in pending if e["priority"] == "critical"]
    high = [e for e in pending if e["priority"] == "high"]

    sites_status = {}
    for site_id, site_data in state.get("sites", {}).items():
        sites_status[site_id] = {
            "arm_state": site_data.get("arm_state", "disarmed"),
            "arm_changed_at": site_data.get("arm_changed_at"),
            "pending_events": len([e for e in pending if e["site_id"] == site_id]),
        }

    return {
        "total_pending": len(pending),
        "critical_count": len(critical),
        "high_count": len(high),
        "sites": sites_status,
        "recent_events": pending[:20],
        "recent_audit": state.get("audit_log", [])[-20:][::-1],
    }


NOTIFICATION_RULES = {
    "critical": {
        "channels": ["push", "sms", "email", "webhook"],
        "delay_seconds": 0,
        "repeat_after_seconds": 60,
        "description": "Immediate alert via all channels",
    },
    "high": {
        "channels": ["push", "email", "webhook"],
        "delay_seconds": 0,
        "repeat_after_seconds": 300,
        "description": "Immediate push + email",
    },
    "medium": {
        "channels": ["push", "webhook"],
        "delay_seconds": 30,
        "repeat_after_seconds": 600,
        "description": "Push notification after 30s delay",
    },
    "low": {
        "channels": ["webhook"],
        "delay_seconds": 60,
        "repeat_after_seconds": 3600,
        "description": "Webhook only, 1min delay",
    },
    "info": {
        "channels": [],
        "delay_seconds": 0,
        "repeat_after_seconds": 0,
        "description": "Logged only, no notification",
    },
}


def get_notification_rules() -> dict[str, Any]:
    return NOTIFICATION_RULES
