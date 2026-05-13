"""Simple session-based authentication for Vigil."""

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

AUTH_FILE = Path("./data/auth.json")
SESSIONS: dict[str, str] = {}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_auth() -> dict[str, Any]:
    if AUTH_FILE.exists():
        return json.loads(AUTH_FILE.read_text())
    return {}


def _save_auth(data: dict[str, Any]) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(data, indent=2))


def is_setup_complete() -> bool:
    """Check if at least one user account exists."""
    auth = _load_auth()
    return len(auth.get("users", [])) > 0


def create_user(username: str, password: str) -> bool:
    """Create a new user account."""
    auth = _load_auth()
    users = auth.get("users", [])

    if any(u["username"] == username for u in users):
        return False

    users.append({"username": username, "password_hash": _hash_password(password)})
    auth["users"] = users
    _save_auth(auth)
    return True


def verify_login(username: str, password: str) -> str | None:
    """Verify credentials and return a session token if valid."""
    auth = _load_auth()
    users = auth.get("users", [])
    pw_hash = _hash_password(password)

    for user in users:
        if user["username"] == username and user["password_hash"] == pw_hash:
            token = secrets.token_urlsafe(32)
            SESSIONS[token] = username
            return token

    return None


def verify_session(token: str | None) -> str | None:
    """Return username if the session token is valid."""
    if not token:
        return None
    return SESSIONS.get(token)


def logout(token: str) -> None:
    """Invalidate a session."""
    SESSIONS.pop(token, None)
