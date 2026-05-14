"""Zone management for Vigil — polygons, counting lines, exclusion areas."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZONES_FILE = Path("./data/zones.json")


@dataclass
class Zone:
    id: str
    name: str
    camera_id: str
    zone_type: str  # "detection", "exclusion", "counting_line", "speed_limit"
    points: list[dict[str, float]]  # [{x: 0-1, y: 0-1}, ...]
    config: dict[str, Any]  # type-specific config (speed_limit, direction, etc.)
    color: str
    enabled: bool = True


def load_zones() -> list[dict[str, Any]]:
    if ZONES_FILE.exists():
        return json.loads(ZONES_FILE.read_text())
    return []


def save_zones(zones: list[dict[str, Any]]) -> None:
    ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ZONES_FILE.write_text(json.dumps(zones, indent=2))


def point_in_polygon(x: float, y: float, polygon: list[dict[str, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["x"], polygon[i]["y"]
        xj, yj = polygon[j]["x"], polygon[j]["y"]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def check_detection_in_zone(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    zone: dict[str, Any],
) -> bool:
    """Check if a detection's center falls within a zone polygon."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / frame_width
    cy = ((y1 + y2) / 2) / frame_height
    return point_in_polygon(cx, cy, zone["points"])


def crosses_line(
    prev_pos: tuple[float, float],
    curr_pos: tuple[float, float],
    line_start: dict[str, float],
    line_end: dict[str, float],
) -> str | None:
    """Check if movement crosses a counting line. Returns 'in', 'out', or None."""
    def ccw(a, b, c):
        return (c["y"]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c["x"]-a[0])

    def ccw2(a, b, c):
        return (c[1]-a["y"])*(b["x"]-a["x"]) > (b["y"]-a["y"])*(c[0]-a["x"])

    a, b = prev_pos, curr_pos
    c, d = line_start, line_end

    a_dict = {"x": a[0], "y": a[1]}
    b_dict = {"x": b[0], "y": b[1]}

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a["x"]-o["x"])*(b["y"]-o["y"]) - (a["y"]-o["y"])*(b["x"]-o["x"])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    if segments_intersect(a_dict, b_dict, c, d):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        line_dx = d["x"] - c["x"]
        line_dy = d["y"] - c["y"]
        cross_product = dx * line_dy - dy * line_dx
        return "in" if cross_product > 0 else "out"

    return None
