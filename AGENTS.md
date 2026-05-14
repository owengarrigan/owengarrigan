# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**cctv-ai-edge** is a single-process Python FastAPI application for local/edge CCTV AI. It runs YOLO person detection on video frames, stores events in SQLite, and serves a Jinja2 dashboard. No external services (databases, caches, queues) are required.

### Running the dev server

```bash
source /workspace/.venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

See `README.md` for full endpoint list and configuration.

### Key endpoints

- `GET /` — HTML dashboard
- `GET /health` — health check
- `GET /events` — recent events JSON
- `GET /events/{id}` — single event
- `GET /snapshots/{filename}` — event snapshot image

### Test video

The detection pipeline needs a video source configured via `CAMERA_SOURCE` in `.env`. A test video at `./data/test-video.mp4` is auto-generated during setup. The YOLO model (`yolov8n.pt`) is auto-downloaded by Ultralytics on first inference.

### Known gotchas

- **Starlette TemplateResponse API**: With unpinned dependencies, the latest Starlette changes the `TemplateResponse` positional argument order. The dashboard endpoint in `app/main.py` uses keyword arguments (`request=`, `name=`, `context=`) to be compatible with the latest version.
- **No automated tests**: The `tests/` directory is empty (`.gitkeep` only). No linting or test framework is configured.
- **No pyproject.toml**: Dependencies are in `requirements.txt` only, with no version pins.
- **OpenCV system deps**: Requires `libgl1` and `libglib2.0-0` on headless Linux (pre-installed in the Cloud Agent VM).
- **`python3.12-venv`**: Required for creating the virtual environment on Ubuntu (pre-installed in the Cloud Agent VM).
- **Video processing is one-shot for files**: When `CAMERA_SOURCE` points to a local `.mp4`, the pipeline processes the entire video once, then the server keeps running so you can inspect events via the dashboard/API.
