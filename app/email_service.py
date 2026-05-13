"""Email notification service for Vigil alerts."""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EMAIL_CONFIG_FILE = Path("./data/email_config.json")


def load_email_config() -> dict[str, Any]:
    if EMAIL_CONFIG_FILE.exists():
        return json.loads(EMAIL_CONFIG_FILE.read_text())
    return {}


def save_email_config(config: dict[str, Any]) -> None:
    EMAIL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMAIL_CONFIG_FILE.write_text(json.dumps(config, indent=2))


def send_email_alert(
    subject: str,
    body_html: str,
    to_email: str | None = None,
) -> bool:
    """Send an email alert using configured SMTP settings."""
    config = load_email_config()
    if not config.get("smtp_host"):
        logger.warning("Email not configured — skipping alert")
        return False

    recipient = to_email or config.get("default_recipient", "")
    if not recipient:
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 Vigil Alert: {subject}"
        msg["From"] = config.get("from_address", f"vigil@{config['smtp_host']}")
        msg["To"] = recipient

        text_body = body_html.replace("<br>", "\n").replace("</p>", "\n")
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        port = config.get("smtp_port", 587)
        use_tls = config.get("use_tls", True)

        if use_tls:
            server = smtplib.SMTP(config["smtp_host"], port, timeout=10)
            server.starttls()
        else:
            server = smtplib.SMTP(config["smtp_host"], port, timeout=10)

        if config.get("smtp_user") and config.get("smtp_password"):
            server.login(config["smtp_user"], config["smtp_password"])

        server.sendmail(msg["From"], [recipient], msg.as_string())
        server.quit()
        logger.info("Email alert sent to %s: %s", recipient, subject)
        return True

    except Exception as e:
        logger.error("Failed to send email: %s", e)
        return False


def format_detection_email(event: dict[str, Any]) -> tuple[str, str]:
    """Format a detection event into email subject and HTML body."""
    event_type = event.get("event_type", "detection").replace("_", " ").title()
    camera = event.get("camera_name", "Unknown")
    confidence = int(event.get("confidence", 0) * 100)
    labels = ", ".join(event.get("labels_detected", []))

    subject = f"{event_type} — {camera}"
    body = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:480px;margin:0 auto;padding:20px">
        <div style="background:#0f1114;border-radius:12px;padding:24px;color:#f1f3f6">
            <h2 style="margin:0 0 16px;font-size:18px;color:#f1f3f6">
                ⚡ {event_type}
            </h2>
            <table style="width:100%;border-collapse:collapse;font-size:14px">
                <tr><td style="padding:8px 0;color:#9ca3af">Camera</td><td style="padding:8px 0;font-weight:600">{camera}</td></tr>
                <tr><td style="padding:8px 0;color:#9ca3af">Confidence</td><td style="padding:8px 0;font-weight:600">{confidence}%</td></tr>
                <tr><td style="padding:8px 0;color:#9ca3af">Labels</td><td style="padding:8px 0;font-weight:600">{labels}</td></tr>
                <tr><td style="padding:8px 0;color:#9ca3af">Time</td><td style="padding:8px 0">{event.get("timestamp","")}</td></tr>
            </table>
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.1);font-size:11px;color:#6b7280">
                Vigil AI Video Intelligence Platform
            </div>
        </div>
    </div>
    """
    return subject, body
