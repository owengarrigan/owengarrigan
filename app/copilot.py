"""Vigil AI Copilot — automatic report generation and insights.

Generates daily/weekly summaries, identifies patterns, and provides
actionable recommendations based on detection history.
"""

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from app.events.event_store import EventStore


def generate_daily_summary(event_store: EventStore) -> dict[str, Any]:
    """Generate an AI-powered daily summary of all activity."""
    events = event_store.get_recent(limit=5000)
    if not events:
        return {"summary": "No activity recorded today.", "insights": [], "stats": {}}

    today = datetime.now(UTC).date()
    today_events = [e for e in events if e.timestamp.startswith(str(today))]

    label_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    hourly: defaultdict[int, int] = defaultdict(int)
    type_counts: Counter[str] = Counter()

    for e in today_events:
        for label in e.labels_detected:
            label_counts[label] += 1
        camera_counts[e.camera_name] += 1
        type_counts[e.event_type] += 1
        try:
            hour = int(e.timestamp[11:13])
            hourly[hour] += 1
        except (ValueError, IndexError):
            pass

    peak_hour = max(hourly, key=hourly.get) if hourly else None
    quietest_hour = min(hourly, key=hourly.get) if hourly else None
    busiest_camera = camera_counts.most_common(1)[0][0] if camera_counts else None

    insights = []

    if peak_hour is not None:
        insights.append({
            "type": "peak_activity",
            "icon": "📈",
            "text": f"Peak activity was at {peak_hour:02d}:00 with {hourly[peak_hour]} detections",
            "priority": "info",
        })

    if busiest_camera:
        insights.append({
            "type": "busiest_camera",
            "icon": "📷",
            "text": f"Busiest camera: {busiest_camera} ({camera_counts[busiest_camera]} events)",
            "priority": "info",
        })

    total = len(today_events)
    people = label_counts.get("person", 0)
    if people > 50:
        insights.append({
            "type": "high_traffic",
            "icon": "⚠️",
            "text": f"High foot traffic: {people} person detections today",
            "priority": "warning",
        })

    animals = sum(label_counts.get(a, 0) for a in ["dog", "cat", "bird", "fox"])
    if animals > 0:
        insights.append({
            "type": "wildlife",
            "icon": "🦊",
            "text": f"Wildlife activity: {animals} animal detections",
            "priority": "info",
        })

    vehicles = sum(label_counts.get(v, 0) for v in ["car", "truck", "bus", "motorcycle"])
    if vehicles > 0:
        insights.append({
            "type": "vehicles",
            "icon": "🚗",
            "text": f"Vehicle activity: {vehicles} detections",
            "priority": "info",
        })

    summary_text = _build_summary_text(total, people, vehicles, animals, peak_hour, busiest_camera)

    return {
        "date": str(today),
        "summary": summary_text,
        "insights": insights,
        "stats": {
            "total_events": total,
            "people": people,
            "vehicles": vehicles,
            "animals": animals,
            "peak_hour": f"{peak_hour:02d}:00" if peak_hour is not None else None,
            "busiest_camera": busiest_camera,
            "cameras_active": len(camera_counts),
            "hourly_breakdown": dict(hourly),
        },
    }


def generate_weekly_summary(event_store: EventStore) -> dict[str, Any]:
    """Generate a weekly summary with trends."""
    events = event_store.get_recent(limit=10000)
    if not events:
        return {"summary": "No activity this week.", "trends": []}

    today = datetime.now(UTC).date()
    week_ago = today - timedelta(days=7)

    daily_counts: defaultdict[str, int] = defaultdict(int)
    daily_labels: defaultdict[str, Counter] = defaultdict(Counter)

    for e in events:
        try:
            event_date = e.timestamp[:10]
            if event_date >= str(week_ago):
                daily_counts[event_date] += 1
                for label in e.labels_detected:
                    daily_labels[event_date][label] += 1
        except (ValueError, IndexError):
            pass

    trends = []
    counts_list = list(daily_counts.values())
    if len(counts_list) >= 2:
        recent_avg = sum(counts_list[-3:]) / min(3, len(counts_list[-3:]))
        older_avg = sum(counts_list[:-3]) / max(1, len(counts_list[:-3]))
        if recent_avg > older_avg * 1.3:
            trends.append({
                "icon": "📈",
                "text": "Activity is trending UP — 30%+ increase in recent days",
                "direction": "up",
            })
        elif recent_avg < older_avg * 0.7:
            trends.append({
                "icon": "📉",
                "text": "Activity is trending DOWN — 30%+ decrease",
                "direction": "down",
            })
        else:
            trends.append({
                "icon": "➡️",
                "text": "Activity is stable this week",
                "direction": "stable",
            })

    total_week = sum(daily_counts.values())
    avg_daily = total_week / max(1, len(daily_counts))

    return {
        "period": f"{week_ago} to {today}",
        "summary": f"Total of {total_week} detections over {len(daily_counts)} days (avg {avg_daily:.0f}/day).",
        "trends": trends,
        "daily_breakdown": dict(daily_counts),
        "total": total_week,
        "average_daily": round(avg_daily),
    }


def get_recommendations(event_store: EventStore) -> list[dict[str, Any]]:
    """Generate actionable recommendations based on patterns."""
    stats = event_store.get_stats()
    recommendations = []

    total = stats.get("total_events", 0)
    if total > 5000:
        recommendations.append({
            "icon": "🗄️",
            "title": "Storage cleanup recommended",
            "description": f"You have {total} events stored. Consider running cleanup to keep the database fast.",
            "action": "cleanup",
            "priority": "medium",
        })

    label_counts = stats.get("label_counts", {})
    if label_counts.get("person", 0) > 100 and not any(
        label_counts.get(a, 0) for a in ["dog", "cat", "bird"]
    ):
        recommendations.append({
            "icon": "🎯",
            "title": "Consider training a custom model",
            "description": "You have 100+ person detections. Fine-tuning could improve accuracy for your specific cameras.",
            "action": "train",
            "priority": "low",
        })

    if total > 0 and stats.get("average_confidence", 0) < 0.4:
        recommendations.append({
            "icon": "⚙️",
            "title": "Low average confidence",
            "description": "Average confidence is below 40%. Consider raising thresholds to reduce false positives.",
            "action": "settings",
            "priority": "high",
        })

    return recommendations


def _build_summary_text(
    total: int, people: int, vehicles: int, animals: int,
    peak_hour: int | None, busiest_camera: str | None,
) -> str:
    """Build a human-readable daily summary paragraph."""
    parts = []
    parts.append(f"Today saw {total} total detections across all cameras.")

    if people > 0:
        parts.append(f"{people} person detections were recorded.")
    if vehicles > 0:
        parts.append(f"{vehicles} vehicle movements were tracked.")
    if animals > 0:
        parts.append(f"{animals} animal sightings were captured.")
    if peak_hour is not None:
        parts.append(f"Peak activity was at {peak_hour:02d}:00.")
    if busiest_camera:
        parts.append(f"Most active camera: {busiest_camera}.")

    return " ".join(parts)
