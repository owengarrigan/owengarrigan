"""Notification delivery system for Vigil alerts."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NOTIFICATION_LOG: list[dict[str, Any]] = []


async def send_webhook(url: str, payload: dict[str, Any]) -> bool:
    """Deliver an alert payload to a webhook URL."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Vigil/1.0"},
            )
            success = response.status_code < 400
            log_notification("webhook", url, payload, success, response.status_code)
            return success
    except Exception as e:
        logger.error("Webhook delivery failed: %s", e)
        log_notification("webhook", url, payload, False, error=str(e))
        return False


async def send_slack(webhook_url: str, event: dict[str, Any]) -> bool:
    """Send a formatted Slack notification."""
    blocks = {
        "text": f"🚨 Vigil Alert: {event.get('event_type', 'Detection')}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{event.get('event_type', 'Detection').replace('_', ' ').title()}*\n"
                            f"Camera: {event.get('camera_name', 'Unknown')}\n"
                            f"Confidence: {int(event.get('confidence', 0) * 100)}%\n"
                            f"Labels: {', '.join(event.get('labels_detected', []))}"
                }
            }
        ]
    }
    return await send_webhook(webhook_url, blocks)


def log_notification(
    channel: str,
    destination: str,
    payload: dict[str, Any],
    success: bool,
    status_code: int | None = None,
    error: str | None = None,
) -> None:
    """Log notification delivery for audit trail."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "channel": channel,
        "destination": destination,
        "success": success,
        "status_code": status_code,
        "error": error,
    }
    NOTIFICATION_LOG.append(entry)
    if len(NOTIFICATION_LOG) > 500:
        NOTIFICATION_LOG.pop(0)


def get_notification_log(limit: int = 50) -> list[dict[str, Any]]:
    return NOTIFICATION_LOG[-limit:][::-1]
