"""Scheduling engine for Vigil alert rules — business hours, quiet hours, custom schedules."""

from datetime import datetime, time
from typing import Any


def is_within_schedule(schedule: dict[str, Any], now: datetime | None = None) -> bool:
    """Check if the current time falls within a schedule definition.

    Schedule format:
    {
        "type": "always" | "business_hours" | "after_hours" | "custom",
        "start_time": "09:00",  # for business_hours/custom
        "end_time": "17:00",    # for business_hours/custom
        "days": [0,1,2,3,4],    # 0=Monday, 6=Sunday (for custom)
        "timezone": "Europe/London"
    }
    """
    if now is None:
        now = datetime.now()

    schedule_type = schedule.get("type", "always")

    if schedule_type == "always":
        return True

    current_time = now.time()
    current_day = now.weekday()

    if schedule_type == "business_hours":
        start = _parse_time(schedule.get("start_time", "09:00"))
        end = _parse_time(schedule.get("end_time", "17:00"))
        days = schedule.get("days", [0, 1, 2, 3, 4])
        return current_day in days and start <= current_time <= end

    if schedule_type == "after_hours":
        start = _parse_time(schedule.get("start_time", "17:00"))
        end = _parse_time(schedule.get("end_time", "09:00"))
        if start > end:
            return current_time >= start or current_time <= end
        return start <= current_time <= end

    if schedule_type == "custom":
        start = _parse_time(schedule.get("start_time", "00:00"))
        end = _parse_time(schedule.get("end_time", "23:59"))
        days = schedule.get("days", [0, 1, 2, 3, 4, 5, 6])
        if current_day not in days:
            return False
        if start > end:
            return current_time >= start or current_time <= end
        return start <= current_time <= end

    return True


def _parse_time(time_str: str) -> time:
    """Parse HH:MM string to time object."""
    parts = time_str.split(":")
    return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


SCHEDULE_PRESETS = {
    "always": {
        "type": "always",
        "label": "Always Active",
        "description": "Rule fires at any time",
    },
    "business_hours": {
        "type": "business_hours",
        "start_time": "09:00",
        "end_time": "17:00",
        "days": [0, 1, 2, 3, 4],
        "label": "Business Hours",
        "description": "Monday–Friday, 9am–5pm",
    },
    "after_hours": {
        "type": "after_hours",
        "start_time": "17:00",
        "end_time": "09:00",
        "label": "After Hours",
        "description": "Evenings, nights, and weekends",
    },
    "weekends": {
        "type": "custom",
        "start_time": "00:00",
        "end_time": "23:59",
        "days": [5, 6],
        "label": "Weekends Only",
        "description": "Saturday and Sunday, all day",
    },
    "overnight": {
        "type": "custom",
        "start_time": "21:00",
        "end_time": "06:00",
        "days": [0, 1, 2, 3, 4, 5, 6],
        "label": "Overnight",
        "description": "9pm–6am every day",
    },
}


def get_schedule_presets() -> list[dict[str, Any]]:
    """Return available schedule presets."""
    return [{"id": k, **v} for k, v in SCHEDULE_PRESETS.items()]
