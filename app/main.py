import asyncio
import json as json_module
import logging
import shutil
import tempfile
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.auth import create_user, is_setup_complete, logout, verify_login, verify_session
from app.camera.pipeline import CameraConfig, PipelineManager
from app.camera.video_reader import VideoReader
from app.config import Settings, get_settings
from app.detection.yolo_detector import YOLODetector
from app.events.event_store import EventStore, StoredEvent
from app.events.snapshot import save_snapshot
from app.rules.person_present_rule import PersonPresentRule

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "web" / "templates"))


def configure_logging() -> None:
    """Configure readable logs for local edge operation."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def event_to_dict(event: StoredEvent) -> dict[str, Any]:
    """Convert a stored event into API/dashboard friendly data."""

    payload = asdict(event)
    payload["snapshot_filename"] = Path(event.snapshot_path).name
    return payload


def run_detection_pipeline(
    settings: Settings,
    event_store: EventStore,
    stop_event: Event,
) -> None:
    """Read frames, run YOLO periodically, evaluate rules, and store events."""

    logger.info("Starting detection pipeline for camera %s", settings.camera_name)
    try:
        with VideoReader(settings.camera_source) as reader:
            detector = YOLODetector(
                model_path=settings.yolo_model_path,
                confidence_threshold=settings.person_confidence_threshold,
                target_labels={"person"},
            )
            person_rule = PersonPresentRule(settings.person_confidence_threshold)

            for frame_number, frame in reader.frames():
                if stop_event.is_set():
                    logger.info("Detection pipeline stop requested")
                    break

                # Sampling every N frames keeps the MVP lightweight on edge devices.
                if frame_number % settings.detection_interval != 0:
                    continue

                try:
                    detections = detector.detect(frame)
                    rule_event = person_rule.evaluate(detections)
                    if rule_event is None:
                        continue

                    snapshot_path = save_snapshot(
                        frame=frame,
                        detections=rule_event.detections,
                        output_dir=settings.snapshots_dir,
                        camera_name=settings.camera_name,
                    )
                    stored_event = event_store.add_event(
                        timestamp=datetime.now(UTC).isoformat(),
                        camera_name=settings.camera_name,
                        event_type=rule_event.event_type,
                        confidence=rule_event.confidence,
                        snapshot_path=str(snapshot_path),
                        labels_detected=rule_event.labels_detected,
                    )
                    logger.info(
                        "Stored event %s type=%s confidence=%.2f snapshot=%s",
                        stored_event.id,
                        stored_event.event_type,
                        stored_event.confidence,
                        stored_event.snapshot_path,
                    )
                except Exception:
                    logger.exception("Failed to process frame %d", frame_number)
    except Exception:
        logger.exception("Detection pipeline stopped because the source failed")
    finally:
        logger.info("Detection pipeline exited")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.ensure_local_storage()

    event_store = EventStore(settings.database_path)
    event_store.initialize()

    pipeline_manager = PipelineManager(
        model_path="yolo11m.pt",
        event_store=event_store,
        snapshots_dir=settings.snapshots_dir,
    )

    stop_event = Event()
    pipeline_task = asyncio.create_task(
        asyncio.to_thread(run_detection_pipeline, settings, event_store, stop_event)
    )

    app.state.settings = settings
    app.state.event_store = event_store
    app.state.stop_event = stop_event
    app.state.pipeline_task = pipeline_task
    app.state.pipeline_manager = pipeline_manager

    yield

    stop_event.set()
    pipeline_manager.stop_all()
    if not pipeline_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=5)
        except TimeoutError:
            logger.warning("Detection pipeline did not stop within timeout")


app = FastAPI(title="Vigil", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "web" / "static")), name="static")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "app": settings.app_name}


@app.get("/stats")
def stats(request: Request) -> dict[str, Any]:
    event_store: EventStore = request.app.state.event_store
    return event_store.get_stats()


class LoginInput(BaseModel):
    username: str
    password: str


@app.post("/auth/setup")
def auth_setup(data: LoginInput) -> dict[str, Any]:
    """Create the first admin account (only works if no users exist)."""
    if is_setup_complete():
        raise HTTPException(status_code=400, detail="Setup already complete")
    if not create_user(data.username, data.password):
        raise HTTPException(status_code=400, detail="User already exists")
    token = verify_login(data.username, data.password)
    return {"status": "ok", "token": token, "username": data.username}


@app.post("/auth/login")
def auth_login(data: LoginInput) -> dict[str, Any]:
    """Authenticate and return a session token."""
    token = verify_login(data.username, data.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": token, "username": data.username}


@app.post("/auth/logout")
def auth_logout(request: Request) -> dict[str, str]:
    """Invalidate the current session."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    logout(token)
    return {"status": "ok"}


@app.get("/auth/status")
def auth_status() -> dict[str, Any]:
    """Check if setup is complete."""
    return {"setup_complete": is_setup_complete()}


def parse_natural_language_query(query: str) -> dict[str, Any]:
    """Parse natural language into structured search filters."""
    query_lower = query.lower().strip()
    filters: dict[str, Any] = {}

    people_words = ["person", "people", "someone", "man", "woman", "human", "pedestrian", "visitor", "intruder", "delivery"]
    vehicle_words = ["car", "van", "truck", "vehicle", "bus", "motorcycle", "bike", "forklift", "trailer"]
    animal_words = ["fox", "dog", "cat", "animal", "bird", "horse", "deer", "wildlife", "creature"]

    for w in people_words:
        if w in query_lower:
            filters["event_type"] = "person_present"
            break
    for w in vehicle_words:
        if w in query_lower:
            filters["event_type"] = "vehicle_present"
            filters["query"] = w if w not in ["vehicle"] else None
            break
    for w in animal_words:
        if w in query_lower:
            filters["event_type"] = "animal_present"
            filters["query"] = w if w not in ["animal", "wildlife", "creature"] else None
            break

    if "high confidence" in query_lower or "confident" in query_lower:
        filters["min_confidence"] = 0.7
    elif "low confidence" in query_lower:
        filters["min_confidence"] = 0.2

    if "today" in query_lower or "this morning" in query_lower or "tonight" in query_lower:
        filters["time_hint"] = "today"
    elif "yesterday" in query_lower:
        filters["time_hint"] = "yesterday"
    elif "this week" in query_lower:
        filters["time_hint"] = "week"

    if not filters.get("query"):
        clean = query_lower
        for w in people_words + vehicle_words + animal_words + ["show me", "find", "all", "the", "from", "last", "night", "today", "yesterday", "this week", "with", "high confidence", "low confidence"]:
            clean = clean.replace(w, "")
        clean = clean.strip()
        if clean and len(clean) > 2:
            filters["query"] = clean

    return filters


@app.get("/events/search")
def search_events(
    request: Request,
    q: str | None = Query(default=None),
    camera: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Search and filter events — supports natural language queries."""
    event_store: EventStore = request.app.state.event_store

    parsed = {}
    if q:
        parsed = parse_natural_language_query(q)

    effective_type = event_type or parsed.get("event_type")
    effective_conf = min_confidence or parsed.get("min_confidence")
    effective_query = parsed.get("query") or (q if not parsed.get("event_type") else None)

    events = event_store.search_events(
        query=effective_query, camera=camera, event_type=effective_type,
        min_confidence=effective_conf, limit=limit, offset=offset,
    )

    descriptions = []
    for e in events[:20]:
        desc = _describe_event(e)
        descriptions.append(desc)

    return {
        "results": [event_to_dict(e) for e in events],
        "descriptions": descriptions,
        "count": len(events),
        "cameras": event_store.get_cameras(),
        "event_types": event_store.get_event_types(),
        "parsed_filters": parsed,
    }


def _describe_event(event: StoredEvent) -> str:
    """Generate a human-readable description of an event."""
    labels = event.labels_detected
    conf = int(event.confidence * 100)
    camera = event.camera_name

    if "person" in labels:
        return f"Person detected at {camera} ({conf}% confidence)"
    elif "dog" in labels or "cat" in labels:
        animal = "dog" if "dog" in labels else "cat"
        return f"Animal ({animal}) spotted at {camera} ({conf}% confidence)"
    elif "car" in labels or "truck" in labels:
        vehicle = next((l for l in labels if l in ["car", "truck", "bus", "motorcycle"]), "vehicle")
        return f"{vehicle.title()} detected at {camera} ({conf}% confidence)"
    else:
        return f"{', '.join(labels)} detected at {camera} ({conf}% confidence)"


ALERTS: list[dict[str, Any]] = []


class AlertRule(BaseModel):
    name: str
    trigger: str
    camera: str = ""
    action: str = "notify"
    enabled: bool = True


ALERT_RULES: list[dict[str, Any]] = []


@app.get("/alerts")
def get_alerts(limit: int = Query(default=50)) -> dict[str, Any]:
    """Get recent alerts and rules."""
    return {
        "alerts": ALERTS[-limit:][::-1],
        "rules": ALERT_RULES,
    }


@app.post("/alerts/rules")
def add_alert_rule(rule: AlertRule) -> dict[str, Any]:
    entry = {"id": str(uuid.uuid4())[:8], **rule.model_dump()}
    ALERT_RULES.append(entry)
    return entry


@app.delete("/alerts/rules/{rule_id}")
def delete_alert_rule(rule_id: str) -> dict[str, str]:
    global ALERT_RULES
    ALERT_RULES = [r for r in ALERT_RULES if r["id"] != rule_id]
    return {"status": "deleted"}


@app.post("/events/cleanup")
def cleanup_events(
    request: Request,
    keep: int = Query(default=1000, ge=100, le=50000),
) -> dict[str, Any]:
    """Delete oldest events beyond the retention limit."""
    event_store: EventStore = request.app.state.event_store
    deleted = event_store.cleanup_old_events(keep_count=keep)
    return {"deleted": deleted, "remaining": event_store.get_event_count()}


@app.get("/pipeline/status")
def pipeline_status(request: Request) -> list[dict[str, Any]]:
    """Get status of all running camera pipelines."""
    manager: PipelineManager | None = getattr(request.app.state, "pipeline_manager", None)
    if manager:
        return manager.get_status()
    return []


@app.get("/events")
def list_events(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    event_store: EventStore = request.app.state.event_store
    return [event_to_dict(event) for event in event_store.get_recent(limit=limit)]


@app.get("/events/{event_id}")
def get_event(request: Request, event_id: int) -> dict[str, Any]:
    event_store: EventStore = request.app.state.event_store
    event = event_store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event_to_dict(event)


@app.get("/snapshots/{filename}")
def get_snapshot(request: Request, filename: str) -> FileResponse:
    settings: Settings = request.app.state.settings
    base_dir = settings.snapshots_dir.resolve()
    snapshot_path = (base_dir / filename).resolve()

    if base_dir not in snapshot_path.parents and snapshot_path != base_dir:
        raise HTTPException(status_code=400, detail="Invalid snapshot path")
    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return FileResponse(snapshot_path)


ANALYSIS_DIR = Path("./data/analysis")

PEOPLE_LABELS = {"person"}
VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
TARGET_LABELS = PEOPLE_LABELS | VEHICLE_LABELS | ANIMAL_LABELS


def classify_label(label: str) -> str:
    """Map a YOLO label to a high-level category."""
    if label in PEOPLE_LABELS:
        return "people"
    if label in VEHICLE_LABELS:
        return "vehicles"
    if label in ANIMAL_LABELS:
        return "animals"
    return "other"


ANALYSIS_MODEL = "yolo11m.pt"

DEFAULT_THRESHOLDS = {"people": 0.45, "vehicles": 0.30, "animals": 0.25}


def analyse_video_file(
    video_path: str,
    model_path: str,
    thresholds: dict[str, float] | None = None,
    enabled_categories: set[str] | None = None,
) -> dict[str, Any]:
    """Run YOLO on sampled frames with per-category confidence filtering."""

    thresholds = thresholds or DEFAULT_THRESHOLDS
    enabled_categories = enabled_categories or {"people", "vehicles", "animals"}

    active_labels: set[str] = set()
    if "people" in enabled_categories:
        active_labels |= PEOPLE_LABELS
    if "vehicles" in enabled_categories:
        active_labels |= VEHICLE_LABELS
    if "animals" in enabled_categories:
        active_labels |= ANIMAL_LABELS

    output_dir = ANALYSIS_DIR / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)

    min_threshold = min(thresholds.values()) if thresholds else 0.2
    detector = YOLODetector(
        model_path=ANALYSIS_MODEL,
        confidence_threshold=min_threshold,
        target_labels=active_labels,
    )

    frames_data: list[dict[str, Any]] = []
    object_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    with VideoReader(video_path) as reader:
        total_frames = 0
        for frame_number, frame in reader.frames():
            total_frames += 1
            if frame_number % 10 != 0:
                continue

            raw_detections = detector.detect(frame)

            filtered = []
            for d in raw_detections:
                category = classify_label(d.label)
                cat_threshold = thresholds.get(category, 0.3)
                if d.confidence >= cat_threshold:
                    filtered.append(d)

            det_list = []
            for d in filtered:
                category = classify_label(d.label)
                det_list.append({
                    "label": d.label,
                    "category": category,
                    "confidence": round(d.confidence, 3),
                    "bbox": list(d.bbox),
                })
                object_counter[d.label] += 1
                category_counter[category] += 1

            snapshot_path = save_snapshot(
                frame=frame,
                detections=filtered,
                output_dir=output_dir,
                camera_name="upload",
            )

            frames_data.append({
                "frame_number": frame_number,
                "snapshot_filename": snapshot_path.name,
                "snapshot_dir": str(output_dir),
                "detections": det_list,
            })

    return {
        "total_frames": total_frames,
        "frames_analysed": len(frames_data),
        "object_summary": dict(object_counter),
        "category_summary": dict(category_counter),
        "frames": frames_data,
    }


@app.post("/analyse")
async def analyse_upload(
    request: Request,
    video: UploadFile,
    people_threshold: float = Form(default=0.45),
    vehicles_threshold: float = Form(default=0.30),
    animals_threshold: float = Form(default=0.25),
    detect_people: bool = Form(default=True),
    detect_vehicles: bool = Form(default=True),
    detect_animals: bool = Form(default=True),
) -> dict[str, Any]:
    """Accept a video upload with per-category thresholds."""

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "people": people_threshold,
        "vehicles": vehicles_threshold,
        "animals": animals_threshold,
    }
    enabled = set()
    if detect_people:
        enabled.add("people")
    if detect_vehicles:
        enabled.add("vehicles")
    if detect_animals:
        enabled.add("animals")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(ANALYSIS_DIR)) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(
            analyse_video_file, tmp_path, ANALYSIS_MODEL, thresholds, enabled
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.get("/analysis-snapshots/{filename}")
def get_analysis_snapshot(filename: str) -> FileResponse:
    """Serve annotated frame snapshots generated by video analysis."""

    for subdir in ANALYSIS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / filename
        if candidate.exists():
            return FileResponse(candidate)

    raise HTTPException(status_code=404, detail="Analysis snapshot not found")


CAMERAS_FILE = Path("./data/cameras.json")
CASES_FILE = Path("./data/cases.json")


class CaseInput(BaseModel):
    title: str
    description: str = ""
    event_ids: list[int] = []


class CaseNoteInput(BaseModel):
    note: str


def load_cases() -> list[dict[str, Any]]:
    if CASES_FILE.exists():
        return json_module.loads(CASES_FILE.read_text())
    return []


def save_cases(cases: list[dict[str, Any]]) -> None:
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CASES_FILE.write_text(json_module.dumps(cases, indent=2))


@app.get("/cases")
def list_cases() -> list[dict[str, Any]]:
    return load_cases()


@app.post("/cases")
def create_case(case: CaseInput) -> dict[str, Any]:
    cases = load_cases()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "title": case.title,
        "description": case.description,
        "event_ids": case.event_ids,
        "notes": [],
        "status": "open",
        "created_at": datetime.now(UTC).isoformat(),
    }
    cases.append(entry)
    save_cases(cases)
    return entry


@app.post("/cases/{case_id}/events")
def add_events_to_case(case_id: str, event_ids: list[int]) -> dict[str, Any]:
    cases = load_cases()
    for case in cases:
        if case["id"] == case_id:
            case["event_ids"] = list(set(case["event_ids"] + event_ids))
            save_cases(cases)
            return case
    raise HTTPException(status_code=404, detail="Case not found")


@app.post("/cases/{case_id}/notes")
def add_note_to_case(case_id: str, data: CaseNoteInput) -> dict[str, Any]:
    cases = load_cases()
    for case in cases:
        if case["id"] == case_id:
            case["notes"].append({
                "text": data.note,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            save_cases(cases)
            return case
    raise HTTPException(status_code=404, detail="Case not found")


@app.patch("/cases/{case_id}/status")
def update_case_status(case_id: str, status: str = Query(...)) -> dict[str, Any]:
    cases = load_cases()
    for case in cases:
        if case["id"] == case_id:
            case["status"] = status
            save_cases(cases)
            return case
    raise HTTPException(status_code=404, detail="Case not found")


@app.delete("/cases/{case_id}")
def delete_case(case_id: str) -> dict[str, str]:
    cases = load_cases()
    cases = [c for c in cases if c["id"] != case_id]
    save_cases(cases)
    return {"status": "deleted"}


@app.get("/system/health")
def system_health(request: Request) -> dict[str, Any]:
    """Detailed system health for the status panel."""
    import os
    settings: Settings = request.app.state.settings
    event_store: EventStore = request.app.state.event_store
    manager: PipelineManager | None = getattr(request.app.state, "pipeline_manager", None)

    db_size = 0
    if settings.database_path.exists():
        db_size = settings.database_path.stat().st_size

    snapshots_size = 0
    snapshots_count = 0
    if settings.snapshots_dir.exists():
        for f in settings.snapshots_dir.iterdir():
            if f.is_file():
                snapshots_size += f.stat().st_size
                snapshots_count += 1

    pipelines = manager.get_status() if manager else []
    active_pipelines = sum(1 for p in pipelines if p.get("running"))

    return {
        "status": "online",
        "model": "yolo11m",
        "model_params": "20.1M",
        "database_size_mb": round(db_size / 1024 / 1024, 2),
        "snapshots_count": snapshots_count,
        "snapshots_size_mb": round(snapshots_size / 1024 / 1024, 2),
        "total_events": event_store.get_event_count(),
        "active_pipelines": active_pipelines,
        "pipelines": pipelines,
        "uptime": "running",
    }


class CameraInput(BaseModel):
    name: str
    url: str
    type: str = "unifi"


def load_cameras() -> list[dict[str, Any]]:
    if CAMERAS_FILE.exists():
        return json_module.loads(CAMERAS_FILE.read_text())
    return []


def save_cameras(cameras: list[dict[str, Any]]) -> None:
    CAMERAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAMERAS_FILE.write_text(json_module.dumps(cameras, indent=2))


@app.get("/cameras")
def list_cameras() -> list[dict[str, Any]]:
    return load_cameras()


@app.post("/cameras")
def add_camera(camera: CameraInput) -> dict[str, Any]:
    cameras = load_cameras()
    entry = {"id": str(uuid.uuid4())[:8], "name": camera.name, "url": camera.url, "type": camera.type}
    cameras.append(entry)
    save_cameras(cameras)
    return entry


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str) -> dict[str, str]:
    cameras = load_cameras()
    cameras = [c for c in cameras if c["id"] != camera_id]
    save_cameras(cameras)
    return {"status": "deleted"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    event_store: EventStore = request.app.state.event_store
    events = [event_to_dict(event) for event in event_store.get_recent(limit=25)]
    cameras = load_cameras()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"events": events, "cameras": cameras},
    )
