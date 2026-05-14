"""API key management for Vigil third-party integrations.

Allows generating keys for external systems to access Vigil's API,
with per-key rate limiting and permission scoping.
"""

import hashlib
import json
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

API_KEYS_FILE = Path("./data/api_keys.json")

_rate_limits: dict[str, list[float]] = {}


def load_api_keys() -> list[dict[str, Any]]:
    if API_KEYS_FILE.exists():
        return json.loads(API_KEYS_FILE.read_text())
    return []


def save_api_keys(keys: list[dict[str, Any]]) -> None:
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_KEYS_FILE.write_text(json.dumps(keys, indent=2))


def generate_api_key(
    name: str,
    permissions: list[str] | None = None,
    rate_limit_per_minute: int = 60,
    expires_days: int | None = None,
) -> dict[str, Any]:
    """Generate a new API key."""
    key = f"vigil_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    entry = {
        "id": secrets.token_hex(4),
        "name": name,
        "key_prefix": key[:12] + "...",
        "key_hash": key_hash,
        "permissions": permissions or ["read"],
        "rate_limit_per_minute": rate_limit_per_minute,
        "created_at": datetime.now(UTC).isoformat(),
        "last_used": None,
        "request_count": 0,
        "active": True,
    }

    keys = load_api_keys()
    keys.append(entry)
    save_api_keys(keys)

    return {"key": key, **entry}


def validate_api_key(key: str) -> dict[str, Any] | None:
    """Validate an API key and check rate limits."""
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    keys = load_api_keys()

    for k in keys:
        if k["key_hash"] == key_hash and k["active"]:
            if not _check_rate_limit(k["id"], k["rate_limit_per_minute"]):
                return None

            k["last_used"] = datetime.now(UTC).isoformat()
            k["request_count"] = k.get("request_count", 0) + 1
            save_api_keys(keys)
            return k

    return None


def revoke_api_key(key_id: str) -> bool:
    """Revoke an API key."""
    keys = load_api_keys()
    for k in keys:
        if k["id"] == key_id:
            k["active"] = False
            save_api_keys(keys)
            return True
    return False


def _check_rate_limit(key_id: str, limit_per_minute: int) -> bool:
    """Simple sliding window rate limiter."""
    now = time.time()
    window = 60.0

    if key_id not in _rate_limits:
        _rate_limits[key_id] = []

    _rate_limits[key_id] = [t for t in _rate_limits[key_id] if now - t < window]

    if len(_rate_limits[key_id]) >= limit_per_minute:
        return False

    _rate_limits[key_id].append(now)
    return True


def get_api_key_stats() -> dict[str, Any]:
    """Return stats about API keys."""
    keys = load_api_keys()
    active = [k for k in keys if k["active"]]
    total_requests = sum(k.get("request_count", 0) for k in keys)
    return {
        "total_keys": len(keys),
        "active_keys": len(active),
        "total_requests": total_requests,
        "keys": [{k: v for k, v in key.items() if k != "key_hash"} for key in keys],
    }


PERMISSION_SCOPES = {
    "read": "Read events, detections, and analytics",
    "write": "Create events, upload videos, manage cameras",
    "admin": "Full access including settings and user management",
    "alerts": "Receive and manage alerts only",
    "lpr": "License plate data access",
    "export": "Export events and reports",
}
