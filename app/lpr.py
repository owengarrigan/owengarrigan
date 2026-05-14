"""License Plate Recognition (LPR/ANPR) for Vigil.

Detects vehicles, crops plate region, runs OCR, matches against known plates database.
Supports: alerts on unknown plates, VIP/whitelist, blacklist, visitor logging.
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PLATES_DB_FILE = Path("./data/plates.json")
PLATE_LOG_FILE = Path("./data/plate_log.json")

_ocr_reader = None


def get_ocr_reader():
    """Lazy-load EasyOCR reader for plate text recognition."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def load_plates_db() -> dict[str, Any]:
    """Load the known plates database."""
    if PLATES_DB_FILE.exists():
        return json.loads(PLATES_DB_FILE.read_text())
    return {"plates": [], "groups": ["whitelist", "blacklist", "staff", "visitor", "vip"]}


def save_plates_db(db: dict[str, Any]) -> None:
    PLATES_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLATES_DB_FILE.write_text(json.dumps(db, indent=2))


def add_known_plate(
    plate_number: str,
    owner: str = "",
    group: str = "whitelist",
    notes: str = "",
    vehicle_description: str = "",
) -> dict[str, Any]:
    """Add a plate to the known database."""
    db = load_plates_db()
    normalised = normalise_plate(plate_number)

    for p in db["plates"]:
        if p["plate"] == normalised:
            p["owner"] = owner or p.get("owner", "")
            p["group"] = group
            p["notes"] = notes or p.get("notes", "")
            p["vehicle"] = vehicle_description or p.get("vehicle", "")
            p["updated_at"] = datetime.now(UTC).isoformat()
            save_plates_db(db)
            return p

    entry = {
        "plate": normalised,
        "owner": owner,
        "group": group,
        "notes": notes,
        "vehicle": vehicle_description,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "seen_count": 0,
        "last_seen": None,
    }
    db["plates"].append(entry)
    save_plates_db(db)
    return entry


def remove_known_plate(plate_number: str) -> bool:
    """Remove a plate from the database."""
    db = load_plates_db()
    normalised = normalise_plate(plate_number)
    original_len = len(db["plates"])
    db["plates"] = [p for p in db["plates"] if p["plate"] != normalised]
    if len(db["plates"]) < original_len:
        save_plates_db(db)
        return True
    return False


def lookup_plate(plate_number: str) -> dict[str, Any] | None:
    """Look up a plate in the known database."""
    db = load_plates_db()
    normalised = normalise_plate(plate_number)
    for p in db["plates"]:
        if p["plate"] == normalised:
            return p
    return None


def log_plate_sighting(
    plate_number: str,
    camera_name: str,
    confidence: float,
    snapshot_path: str = "",
) -> dict[str, Any]:
    """Log a plate sighting and update the database if known."""
    normalised = normalise_plate(plate_number)
    known = lookup_plate(normalised)

    if known:
        db = load_plates_db()
        for p in db["plates"]:
            if p["plate"] == normalised:
                p["seen_count"] = p.get("seen_count", 0) + 1
                p["last_seen"] = datetime.now(UTC).isoformat()
                break
        save_plates_db(db)

    entry = {
        "plate": normalised,
        "camera": camera_name,
        "confidence": confidence,
        "snapshot": snapshot_path,
        "timestamp": datetime.now(UTC).isoformat(),
        "known": known is not None,
        "group": known["group"] if known else "unknown",
        "owner": known["owner"] if known else "",
    }

    log = load_plate_log()
    log.append(entry)
    if len(log) > 10000:
        log = log[-10000:]
    save_plate_log(log)

    return entry


def load_plate_log() -> list[dict[str, Any]]:
    if PLATE_LOG_FILE.exists():
        return json.loads(PLATE_LOG_FILE.read_text())
    return []


def save_plate_log(log: list[dict[str, Any]]) -> None:
    PLATE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLATE_LOG_FILE.write_text(json.dumps(log, indent=2))


def normalise_plate(plate: str) -> str:
    """Normalise a plate number: uppercase, remove spaces/special chars."""
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def read_plate_from_image(image: np.ndarray) -> list[dict[str, Any]]:
    """Attempt to read license plate text from an image crop.

    Uses EasyOCR to detect and read text, then filters for plate-like patterns.
    """
    reader = get_ocr_reader()
    results = reader.readtext(image)

    plates_found = []
    for (bbox, text, confidence) in results:
        cleaned = normalise_plate(text)
        if len(cleaned) >= 4 and confidence > 0.3:
            if _looks_like_plate(cleaned):
                plates_found.append({
                    "text": cleaned,
                    "raw_text": text,
                    "confidence": round(confidence, 3),
                    "bbox": bbox,
                })

    return plates_found


def read_plate_from_vehicle_crop(
    frame: np.ndarray,
    vehicle_bbox: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Given a full frame and a vehicle bounding box, try to read the plate.

    Crops the lower portion of the vehicle (where plates typically are)
    and runs OCR on that region.
    """
    x1, y1, x2, y2 = vehicle_bbox
    h = y2 - y1
    plate_region_y1 = y1 + int(h * 0.5)
    plate_region = frame[plate_region_y1:y2, x1:x2]

    if plate_region.size == 0:
        return []

    plate_region_resized = cv2.resize(plate_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_region_resized, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    results = read_plate_from_image(plate_region_resized)

    if not results:
        results = read_plate_from_image(enhanced)

    return results


def _looks_like_plate(text: str) -> bool:
    """Heuristic check if text looks like a license plate.

    UK format: AB12 CDE (2 letters, 2 numbers, 3 letters)
    Irish format: 123-D-4567
    US format: varies by state, typically 5-8 alphanumeric
    """
    if len(text) < 4 or len(text) > 10:
        return False
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    if not (has_letters and has_numbers):
        return False
    uk_pattern = re.match(r"^[A-Z]{2}\d{2}[A-Z]{3}$", text)
    if uk_pattern:
        return True
    irish_pattern = re.match(r"^\d{2,3}[A-Z]{1,2}\d{1,6}$", text)
    if irish_pattern:
        return True
    if 4 <= len(text) <= 8:
        return True
    return False


def get_plate_stats() -> dict[str, Any]:
    """Return LPR statistics."""
    db = load_plates_db()
    log = load_plate_log()

    groups_count: dict[str, int] = {}
    for p in db["plates"]:
        g = p.get("group", "unknown")
        groups_count[g] = groups_count.get(g, 0) + 1

    recent_unknowns = [e for e in log[-100:] if not e.get("known")]

    return {
        "total_known_plates": len(db["plates"]),
        "groups": groups_count,
        "total_sightings": len(log),
        "recent_unknown_count": len(recent_unknowns),
        "available_groups": db.get("groups", []),
    }
