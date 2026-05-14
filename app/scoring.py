"""Activity scoring and risk assessment for Vigil.

Assigns a risk score to each detection based on context:
- Time of day (after hours = higher risk)
- Arm state (armed = higher risk)
- Detection type (person > vehicle > animal in security context)
- Confidence level
- Zone (restricted area = higher risk)
- Frequency (unusual activity = higher risk)
"""

from datetime import UTC, datetime
from typing import Any


def calculate_risk_score(
    event_type: str,
    confidence: float,
    arm_state: str = "disarmed",
    is_after_hours: bool = False,
    in_restricted_zone: bool = False,
    is_anomalous: bool = False,
    is_known_plate: bool = False,
) -> dict[str, Any]:
    """Calculate a 0-100 risk score for a detection event."""
    score = 0.0
    factors: list[dict[str, Any]] = []

    type_scores = {
        "person_present": 40,
        "vehicle_present": 25,
        "animal_present": 10,
        "object_detected": 15,
    }
    base = type_scores.get(event_type, 20)
    score += base
    factors.append({"factor": "detection_type", "value": event_type, "points": base})

    conf_bonus = confidence * 20
    score += conf_bonus
    factors.append({"factor": "confidence", "value": f"{confidence:.0%}", "points": round(conf_bonus, 1)})

    if arm_state in ("armed", "armed_away"):
        score += 25
        factors.append({"factor": "arm_state", "value": arm_state, "points": 25})
    elif arm_state == "armed_home":
        score += 10
        factors.append({"factor": "arm_state", "value": arm_state, "points": 10})

    if is_after_hours:
        score += 15
        factors.append({"factor": "after_hours", "value": True, "points": 15})

    if in_restricted_zone:
        score += 20
        factors.append({"factor": "restricted_zone", "value": True, "points": 20})

    if is_anomalous:
        score += 15
        factors.append({"factor": "anomaly", "value": True, "points": 15})

    if is_known_plate:
        score -= 20
        factors.append({"factor": "known_plate", "value": True, "points": -20})

    score = max(0, min(100, score))

    if score >= 80:
        level = "critical"
    elif score >= 60:
        level = "high"
    elif score >= 40:
        level = "medium"
    elif score >= 20:
        level = "low"
    else:
        level = "minimal"

    return {
        "score": round(score),
        "level": level,
        "factors": factors,
        "recommendation": _get_recommendation(level, event_type),
    }


def _get_recommendation(level: str, event_type: str) -> str:
    """Generate a recommendation based on risk level."""
    if level == "critical":
        return "Immediate investigation required. Consider contacting authorities."
    if level == "high":
        return "Review immediately. Check live feed and recent activity."
    if level == "medium":
        return "Monitor situation. Review when convenient."
    if level == "low":
        return "Routine detection. No action needed."
    return "Normal activity within expected parameters."
