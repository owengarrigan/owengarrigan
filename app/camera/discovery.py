"""Camera discovery and brand presets for Vigil."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CameraPreset:
    """RTSP URL template for a camera brand."""
    brand: str
    name: str
    url_template: str
    default_port: int
    notes: str


CAMERA_PRESETS: dict[str, CameraPreset] = {
    "hikvision": CameraPreset(
        brand="Hikvision",
        name="Hikvision (DS-2CD / DS-76xx / DS-2DE)",
        url_template="rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        default_port=554,
        notes="Channel 101=main stream CH1, 102=sub stream CH1, 201=main CH2",
    ),
    "hikvision_sub": CameraPreset(
        brand="Hikvision",
        name="Hikvision Sub-stream (lower bandwidth)",
        url_template="rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}02",
        default_port=554,
        notes="Sub-stream is lower resolution, ideal for AI processing",
    ),
    "unifi_rtsps": CameraPreset(
        brand="UniFi Protect",
        name="UniFi Protect (RTSPS encrypted)",
        url_template="rtsps://{ip}:7441/{token}?enableSrtp",
        default_port=7441,
        notes="Get token from UniFi Protect → Camera → Advanced → RTSP",
    ),
    "unifi_rtsp": CameraPreset(
        brand="UniFi Protect",
        name="UniFi Protect (RTSP unencrypted)",
        url_template="rtsp://{ip}:7447/{token}",
        default_port=7447,
        notes="Enable RTSP in UniFi Protect → Camera → Advanced",
    ),
    "dahua": CameraPreset(
        brand="Dahua",
        name="Dahua / Amcrest",
        url_template="rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        default_port=554,
        notes="subtype=0 is main stream, subtype=1 is sub stream",
    ),
    "reolink": CameraPreset(
        brand="Reolink",
        name="Reolink",
        url_template="rtsp://{username}:{password}@{ip}:{port}/h264Preview_{channel}_main",
        default_port=554,
        notes="Use h264Preview for H.264 compatibility, channel starts at 01",
    ),
    "axis": CameraPreset(
        brand="Axis",
        name="Axis Communications",
        url_template="rtsp://{username}:{password}@{ip}:{port}/axis-media/media.amp",
        default_port=554,
        notes="Standard Axis RTSP path for all models",
    ),
    "generic_onvif": CameraPreset(
        brand="ONVIF",
        name="Generic ONVIF Camera",
        url_template="rtsp://{username}:{password}@{ip}:{port}/onvif/profile{channel}/media.smp",
        default_port=554,
        notes="Standard ONVIF media profile path",
    ),
    "scrypted": CameraPreset(
        brand="Scrypted",
        name="Scrypted (HomeKit / Ring / Nest)",
        url_template="rtsp://{ip}:{port}/{camera_name}",
        default_port=10554,
        notes="Scrypted rebroadcasts any camera as RTSP on localhost",
    ),
}


def get_presets_list() -> list[dict[str, Any]]:
    """Return all presets as serializable dicts."""
    return [
        {
            "id": key,
            "brand": preset.brand,
            "name": preset.name,
            "url_template": preset.url_template,
            "default_port": preset.default_port,
            "notes": preset.notes,
        }
        for key, preset in CAMERA_PRESETS.items()
    ]


def build_rtsp_url(preset_id: str, params: dict[str, str]) -> str:
    """Build an RTSP URL from a preset and user parameters."""
    preset = CAMERA_PRESETS.get(preset_id)
    if not preset:
        raise ValueError(f"Unknown preset: {preset_id}")

    url = preset.url_template
    defaults = {
        "port": str(preset.default_port),
        "channel": "1",
        "username": "admin",
        "password": "admin",
        "token": "",
        "camera_name": "camera",
    }
    defaults.update(params)

    for key, value in defaults.items():
        url = url.replace(f"{{{key}}}", value)

    return url
