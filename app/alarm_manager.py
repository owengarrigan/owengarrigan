"""Alarm Manager for Vigil — UniFi Protect-style trigger → scope → schedule → action.

Each alarm rule consists of:
- Trigger: what event fires the alarm (detection type, zone entry, speed, plate, etc.)
- Scope: which cameras/zones/sites it applies to
- Schedule: when the rule is active (always, business hours, custom)
- Actions: what to do when triggered (notify, webhook, hardware, escalate)
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALARM_RULES_FILE = Path("./data/alarm_rules.json")


TRIGGER_TYPES = {
    "person_detected": {
        "label": "Person Detected",
        "description": "Fires when a person is detected by AI",
        "icon": "👤",
        "category": "detection",
    },
    "vehicle_detected": {
        "label": "Vehicle Detected",
        "description": "Fires when a vehicle (car, truck, van) is detected",
        "icon": "🚗",
        "category": "detection",
    },
    "animal_detected": {
        "label": "Animal Detected",
        "description": "Fires when an animal is detected",
        "icon": "🦊",
        "category": "detection",
    },
    "any_motion": {
        "label": "Any Motion",
        "description": "Fires on any detection regardless of type",
        "icon": "⚡",
        "category": "detection",
    },
    "zone_entry": {
        "label": "Zone Entry",
        "description": "Fires when an object enters a specific zone",
        "icon": "🚧",
        "category": "zone",
    },
    "zone_exit": {
        "label": "Zone Exit",
        "description": "Fires when an object leaves a specific zone",
        "icon": "🚪",
        "category": "zone",
    },
    "speed_violation": {
        "label": "Speed Violation",
        "description": "Fires when a vehicle exceeds the zone speed limit",
        "icon": "🏎️",
        "category": "zone",
    },
    "dwell_exceeded": {
        "label": "Dwell Time Exceeded",
        "description": "Fires when someone remains in a zone too long",
        "icon": "⏱️",
        "category": "zone",
    },
    "plate_unknown": {
        "label": "Unknown Plate",
        "description": "Fires when an unrecognised plate is detected",
        "icon": "🔢",
        "category": "lpr",
    },
    "plate_blacklist": {
        "label": "Blacklisted Plate",
        "description": "Fires when a blacklisted vehicle is detected",
        "icon": "🚫",
        "category": "lpr",
    },
    "occupancy_high": {
        "label": "High Occupancy",
        "description": "Fires when zone occupancy exceeds threshold",
        "icon": "👥",
        "category": "occupancy",
    },
    "anomaly": {
        "label": "Anomaly Detected",
        "description": "Fires when activity is significantly above/below normal",
        "icon": "📊",
        "category": "intelligence",
    },
}

ACTION_TYPES = {
    "push_notification": {
        "label": "Push Notification",
        "description": "Send push notification to mobile devices",
        "icon": "📱",
    },
    "email": {
        "label": "Email Alert",
        "description": "Send email to configured recipients",
        "icon": "📧",
    },
    "webhook": {
        "label": "Webhook",
        "description": "POST to external URL (Slack, Discord, IFTTT, etc.)",
        "icon": "🔗",
    },
    "sms": {
        "label": "SMS Alert",
        "description": "Send SMS to configured phone numbers",
        "icon": "💬",
    },
    "escalate": {
        "label": "Escalate to Monitoring",
        "description": "Add to monitoring station queue for operator review",
        "icon": "🚨",
    },
    "save_clip": {
        "label": "Save Video Clip",
        "description": "Save a 30-second clip around the event",
        "icon": "🎬",
    },
    "arm_system": {
        "label": "Arm System",
        "description": "Automatically arm the site",
        "icon": "🔒",
    },
}

SCHEDULE_TYPES = {
    "always": {"label": "Always", "description": "Rule is active 24/7"},
    "business_hours": {"label": "Business Hours", "description": "Mon-Fri 9am-5pm"},
    "after_hours": {"label": "After Hours", "description": "Outside business hours"},
    "overnight": {"label": "Overnight", "description": "9pm-6am daily"},
    "weekends": {"label": "Weekends", "description": "Saturday & Sunday"},
    "custom": {"label": "Custom", "description": "Define your own schedule"},
}


def load_alarm_rules() -> list[dict[str, Any]]:
    if ALARM_RULES_FILE.exists():
        return json.loads(ALARM_RULES_FILE.read_text())
    return []


def save_alarm_rules(rules: list[dict[str, Any]]) -> None:
    ALARM_RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALARM_RULES_FILE.write_text(json.dumps(rules, indent=2))


def create_alarm_rule(
    name: str,
    trigger: str,
    scope: dict[str, Any],
    schedule: str,
    actions: list[str],
    config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Create a new alarm rule."""
    import uuid
    rules = load_alarm_rules()
    rule = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "trigger": trigger,
        "scope": scope,
        "schedule": schedule,
        "actions": actions,
        "config": config or {},
        "enabled": enabled,
        "created_at": datetime.now(UTC).isoformat(),
        "last_triggered": None,
        "trigger_count": 0,
    }
    rules.append(rule)
    save_alarm_rules(rules)
    return rule


def update_alarm_rule(rule_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    rules = load_alarm_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule.update(updates)
            save_alarm_rules(rules)
            return rule
    return None


def delete_alarm_rule(rule_id: str) -> bool:
    rules = load_alarm_rules()
    original = len(rules)
    rules = [r for r in rules if r["id"] != rule_id]
    if len(rules) < original:
        save_alarm_rules(rules)
        return True
    return False


def toggle_alarm_rule(rule_id: str) -> dict[str, Any] | None:
    rules = load_alarm_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_alarm_rules(rules)
            return rule
    return None


def get_alarm_manager_config() -> dict[str, Any]:
    """Return the full alarm manager configuration for the UI."""
    return {
        "trigger_types": TRIGGER_TYPES,
        "action_types": ACTION_TYPES,
        "schedule_types": SCHEDULE_TYPES,
        "rules": load_alarm_rules(),
    }
