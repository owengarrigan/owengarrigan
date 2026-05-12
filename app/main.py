import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

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

    stop_event = Event()
    pipeline_task = asyncio.create_task(
        asyncio.to_thread(run_detection_pipeline, settings, event_store, stop_event)
    )

    app.state.settings = settings
    app.state.event_store = event_store
    app.state.stop_event = stop_event
    app.state.pipeline_task = pipeline_task

    yield

    stop_event.set()
    if not pipeline_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=5)
        except TimeoutError:
            logger.warning("Detection pipeline did not stop within timeout")


app = FastAPI(title="cctv-ai-edge", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "app": settings.app_name}


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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    event_store: EventStore = request.app.state.event_store
    events = [event_to_dict(event) for event in event_store.get_recent(limit=25)]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"events": events},
    )
