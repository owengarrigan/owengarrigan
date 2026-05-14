"""Multi-site management for Vigil.

Supports multiple physical locations, each with their own cameras,
zones, rules, and analytics — all managed from one dashboard.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SITES_FILE = Path("./data/sites.json")


def load_sites() -> list[dict[str, Any]]:
    if SITES_FILE.exists():
        return json.loads(SITES_FILE.read_text())
    return []


def save_sites(sites: list[dict[str, Any]]) -> None:
    SITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SITES_FILE.write_text(json.dumps(sites, indent=2))


def add_site(
    name: str,
    address: str = "",
    timezone: str = "Europe/Dublin",
    industry: str = "security",
) -> dict[str, Any]:
    """Create a new site/location."""
    sites = load_sites()
    import uuid
    site = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "address": address,
        "timezone": timezone,
        "industry": industry,
        "cameras": [],
        "created_at": datetime.now(UTC).isoformat(),
        "status": "active",
    }
    sites.append(site)
    save_sites(sites)
    return site


def get_site(site_id: str) -> dict[str, Any] | None:
    sites = load_sites()
    return next((s for s in sites if s["id"] == site_id), None)


def update_site(site_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    sites = load_sites()
    for site in sites:
        if site["id"] == site_id:
            site.update(updates)
            save_sites(sites)
            return site
    return None


def delete_site(site_id: str) -> bool:
    sites = load_sites()
    original = len(sites)
    sites = [s for s in sites if s["id"] != site_id]
    if len(sites) < original:
        save_sites(sites)
        return True
    return False


def get_site_summary() -> dict[str, Any]:
    """Return a summary across all sites."""
    sites = load_sites()
    return {
        "total_sites": len(sites),
        "active_sites": sum(1 for s in sites if s.get("status") == "active"),
        "industries": list(set(s.get("industry", "other") for s in sites)),
        "sites": sites,
    }


INDUSTRY_TEMPLATES = {
    "security": {
        "name": "Security",
        "modules": ["intrusion", "perimeter", "after_hours"],
        "default_rules": [
            {"name": "After-hours motion", "trigger": "any_detection", "schedule": "after_hours"},
            {"name": "Perimeter breach", "trigger": "person_in_zone", "schedule": "always"},
        ],
    },
    "retail": {
        "name": "Retail",
        "modules": ["footfall", "heatmap", "dwell_time", "queue"],
        "default_rules": [
            {"name": "High occupancy", "trigger": "occupancy_above", "threshold": 50},
            {"name": "Long queue", "trigger": "queue_length", "threshold": 8},
        ],
    },
    "warehouse": {
        "name": "Warehouse & Fleet",
        "modules": ["speed_limit", "zone_violation", "ppe", "vehicle_tracking"],
        "default_rules": [
            {"name": "Speed violation", "trigger": "speed_above", "threshold": 5},
            {"name": "No PPE detected", "trigger": "missing_ppe", "schedule": "business_hours"},
        ],
    },
    "events": {
        "name": "Events & Logistics",
        "modules": ["vehicle_counting", "bay_occupancy", "crew_count", "turnaround"],
        "default_rules": [
            {"name": "Bay occupied > 1hr", "trigger": "dwell_exceeded", "threshold": 3600},
            {"name": "Load-in complete", "trigger": "no_vehicles", "schedule": "custom"},
        ],
    },
    "leisure": {
        "name": "Leisure & Sports",
        "modules": ["visitor_counting", "peak_hours", "after_hours_security"],
        "default_rules": [
            {"name": "After-hours entry", "trigger": "any_detection", "schedule": "after_hours"},
            {"name": "High usage", "trigger": "occupancy_above", "threshold": 30},
        ],
    },
}


def get_industry_templates() -> dict[str, Any]:
    return INDUSTRY_TEMPLATES
