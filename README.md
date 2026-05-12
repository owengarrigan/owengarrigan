# cctv-ai-edge

Python MVP for a modular, local-first edge CCTV AI platform.

The first version connects to one RTSP stream or local video file, runs
Ultralytics YOLO person detection every configurable number of frames, creates a
local event when a person is present, saves an annotated snapshot, stores event
metadata in SQLite, and exposes a small FastAPI dashboard.

## MVP features

- FastAPI backend with JSON endpoints and a simple local web dashboard.
- OpenCV video input for RTSP URLs and local video files.
- Ultralytics YOLO inference using a standard COCO model such as `yolov8n.pt`.
- Person-present rule with configurable confidence threshold.
- SQLite event storage at `./data/events.db`.
- Annotated event snapshots saved under `./data/events`.
- Clean module boundaries for future plugins such as PPE, animals, sports,
  security, and vehicles.
- Local-only operation. No authentication, cloud upload, or Docker required.

## Project structure

```text
cctv-ai-edge/
  app/
    main.py
    config.py
    camera/
      video_reader.py
    detection/
      base_detector.py
      yolo_detector.py
    rules/
      base_rule.py
      person_present_rule.py
    events/
      event_store.py
      snapshot.py
    plugins/
      base_plugin.py
      ppe_plugin.py
    web/
      templates/
        index.html
  data/
    events/
  models/
  tests/
  .env.example
  requirements.txt
  README.md
```

## Setup

Use Python 3.11 or newer. These steps work locally on macOS, including Apple
Silicon, and on Linux.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first YOLO run downloads the configured model, for example `yolov8n.pt`, if
it is not already present.

### 3. Configure a camera or test video

Copy the example environment file:

```bash
cp .env.example .env
```

Set `CAMERA_SOURCE` in `.env` to a local test video:

```env
CAMERA_SOURCE=/absolute/path/to/test-video.mp4
CAMERA_NAME=front-gate
DETECTION_INTERVAL=10
PERSON_CONFIDENCE_THRESHOLD=0.5
```

You can also use an RTSP stream:

```env
CAMERA_SOURCE=rtsp://user:password@192.168.1.10:554/stream1
```

## Run the FastAPI server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the local dashboard:

```text
http://localhost:8000/
```

Health check:

```text
http://localhost:8000/health
```

Recent events:

```text
http://localhost:8000/events
```

The detection pipeline starts with the FastAPI app. For local video files, it
processes the video once and leaves the API/dashboard running so you can inspect
stored events.

## API endpoints

- `GET /health` - service status.
- `GET /events` - recent events, newest first.
- `GET /events/{id}` - one event by ID.
- `GET /snapshots/{filename}` - event snapshot image.
- `GET /` - dashboard with recent event thumbnails.

## Configuration

Environment variables are loaded from `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_SOURCE` | `./data/test-video.mp4` | RTSP URL or local video file path. |
| `CAMERA_NAME` | `default-camera` | Name stored with each event. |
| `DETECTION_INTERVAL` | `10` | Run inference every N frames. |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Ultralytics model path/name. |
| `PERSON_CONFIDENCE_THRESHOLD` | `0.5` | Minimum person confidence for events. |
| `DATABASE_PATH` | `./data/events.db` | Local SQLite database path. |
| `SNAPSHOTS_DIR` | `./data/events` | Folder for event snapshots. |
| `APP_HOST` | `0.0.0.0` | Optional host setting for custom launch scripts. |
| `APP_PORT` | `8000` | Optional port setting for custom launch scripts. |

## How the pipeline works

1. FastAPI starts and initializes local storage.
2. A background pipeline opens `CAMERA_SOURCE` with OpenCV.
3. Every `DETECTION_INTERVAL` frames, YOLO runs inference.
4. `PersonPresentRule` checks for `person` detections above the threshold.
5. When the rule triggers, the app saves an annotated snapshot.
6. SQLite stores timestamp, camera, event type, confidence, snapshot path, and
   detected labels.
7. The dashboard and API read events from SQLite.

## Future plugin design

The MVP keeps detectors, rules, and plugins separate:

- `BaseDetector` normalizes model output.
- `BaseRule` turns detections into events.
- `BasePlugin` can bundle a detector and rules for a domain.
- `PPEPlugin` is a placeholder showing how to swap a PPE YOLO model later.

Future plugins can follow the same shape:

- `PPEPlugin`: hardhat and hi-vis detection.
- `AnimalPlugin`: dog, horse, and livestock detection.
- `SportPlugin`: player and ball tracking.
- `SecurityPlugin`: loitering and intrusion zones.
- `VehiclePlugin`: cars, vans, speed, and gate events.

## Notes

- This MVP is intentionally local-first and edge-first.
- There is no authentication yet.
- There is no cloud upload yet.
- There is no Docker setup yet.
- The dashboard is deliberately simple HTML rendered by FastAPI/Jinja2.
