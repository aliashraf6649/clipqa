# Run YOLO with a tracker over a clip and return a ClipQA track table.

from pathlib import Path

import cv2
import polars as pl
from ultralytics import YOLO

from clipqa.tracks import TRACK_SCHEMA

# COCO classes ClipQA cares about: person, bicycle, car, motorcycle, bus, truck.
CLASSES = [0, 1, 2, 3, 5, 7]

def track_clip(video: Path, model_path: str = "yolo26m.pt", imgsz: int = 1280, tracker: str = "bytetrack.yaml") -> pl.DataFrame:
    # Detect and track objects in a clip, returning a track table.
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    model = YOLO(model_path)
    results = model.track(source = str(video), stream = True, tracker = tracker, 
                          classes = CLASSES, imgsz = imgsz, device = 0, verbose = False)

    rows = []
    for frame, result in enumerate(results):
        boxes = result.boxes
        if boxes.id is None:
            continue
        for track_id, cls, conf, (x1, y1, x2, y2) in zip(
            boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            rows.append({"frame": frame, "t": frame / fps, "track_id": track_id, "label": result.names[cls],
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
    return pl.DataFrame(rows, schema=TRACK_SCHEMA).sort("frame", "track_id")
        