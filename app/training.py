"""Custom model training pipeline for Vigil.

Allows users to correct detection labels, build a training dataset
from their own cameras, and trigger fine-tuning on Apple Silicon (MPS).
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRAINING_DIR = Path("./data/training")
CORRECTIONS_FILE = TRAINING_DIR / "corrections.json"
DATASET_DIR = TRAINING_DIR / "dataset"


def ensure_training_dirs() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images").mkdir(exist_ok=True)
    (DATASET_DIR / "labels").mkdir(exist_ok=True)


def load_corrections() -> list[dict[str, Any]]:
    if CORRECTIONS_FILE.exists():
        return json.loads(CORRECTIONS_FILE.read_text())
    return []


def save_corrections(corrections: list[dict[str, Any]]) -> None:
    ensure_training_dirs()
    CORRECTIONS_FILE.write_text(json.dumps(corrections, indent=2))


def add_correction(
    event_id: int,
    snapshot_path: str,
    original_label: str,
    correct_label: str,
    bbox: list[int],
    frame_width: int = 640,
    frame_height: int = 480,
) -> dict[str, Any]:
    """Record a label correction from a user.

    This builds the training dataset over time. When enough corrections
    accumulate, the model can be fine-tuned.
    """
    corrections = load_corrections()
    entry = {
        "id": len(corrections) + 1,
        "event_id": event_id,
        "snapshot_path": snapshot_path,
        "original_label": original_label,
        "correct_label": correct_label,
        "bbox": bbox,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    corrections.append(entry)
    save_corrections(corrections)
    return entry


def get_training_stats() -> dict[str, Any]:
    """Return statistics about the training dataset."""
    corrections = load_corrections()
    label_counts: dict[str, int] = {}
    for c in corrections:
        label = c["correct_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_corrections": len(corrections),
        "label_counts": label_counts,
        "ready_to_train": len(corrections) >= 50,
        "recommended_minimum": 50,
        "dataset_path": str(DATASET_DIR),
    }


def build_yolo_dataset() -> dict[str, Any]:
    """Convert corrections into YOLO training format.

    Creates:
    - dataset/images/ — copies of snapshot images
    - dataset/labels/ — YOLO format .txt files (class cx cy w h)
    - dataset/data.yaml — dataset configuration
    """
    ensure_training_dirs()
    corrections = load_corrections()
    if not corrections:
        return {"status": "empty", "images": 0}

    labels_set: set[str] = set()
    for c in corrections:
        labels_set.add(c["correct_label"])
    labels_list = sorted(labels_set)
    label_to_idx = {l: i for i, l in enumerate(labels_list)}

    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"

    for f in images_dir.glob("*"):
        f.unlink()
    for f in labels_dir.glob("*"):
        f.unlink()

    processed = 0
    for c in corrections:
        src = Path(c["snapshot_path"])
        if not src.exists():
            continue

        img_name = f"img_{c['id']:04d}{src.suffix}"
        dst = images_dir / img_name
        shutil.copy2(src, dst)

        x1, y1, x2, y2 = c["bbox"]
        fw, fh = c["frame_width"], c["frame_height"]
        cx = ((x1 + x2) / 2) / fw
        cy = ((y1 + y2) / 2) / fh
        w = (x2 - x1) / fw
        h = (y2 - y1) / fh
        cls_idx = label_to_idx[c["correct_label"]]

        label_file = labels_dir / f"img_{c['id']:04d}.txt"
        with open(label_file, "a") as f:
            f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        processed += 1

    yaml_content = f"""# Vigil Custom Training Dataset
# Auto-generated from user corrections
# {datetime.now(UTC).isoformat()}

path: {DATASET_DIR.resolve()}
train: images
val: images

names:
"""
    for i, label in enumerate(labels_list):
        yaml_content += f"  {i}: {label}\n"

    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    return {
        "status": "ready",
        "images": processed,
        "labels": labels_list,
        "dataset_yaml": str(DATASET_DIR / "data.yaml"),
    }


def get_training_command() -> dict[str, str]:
    """Return the command to run fine-tuning on Apple Silicon."""
    return {
        "command": f"yolo detect train data={DATASET_DIR.resolve()}/data.yaml model=yolo11m.pt epochs=50 device=mps imgsz=640 project=./models/custom name=vigil_custom",
        "python_code": f"""from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="{DATASET_DIR.resolve()}/data.yaml",
    epochs=50,
    device="mps",  # Apple Silicon GPU
    imgsz=640,
    project="./models/custom",
    name="vigil_custom",
)
""",
        "estimated_time": "1-2 hours on M4 Mac Mini (50-200 images)",
        "notes": "Run on the Mac Mini. Training uses Metal Performance Shaders (MPS) for GPU acceleration.",
    }
