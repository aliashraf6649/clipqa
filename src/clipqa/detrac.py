# Load UA-DETRAC ground truth as a ClipQA track table.
import xml.etree.cElementTree as ET
from pathlib import Path

import polars as pl

from clipqa.tracks import TRACK_SCHEMA

FPS = 25.0  # every UA-DETRAC video runs at 25 frames per second

# UA-DETRAC vehicle types, renamed to the COCO labels YOLO uses. COCO has no "van"
# (YOLO usually calls one a car), and "others" are mostly trucks.

TYPE_TO_LABEL = {"car": "car", "bus": "bus", "van": "car", "others": "truck"}

def load_detrac_gt(xml_path: Path) -> pl.DataFrame:
    # Read one UA-DETRAC sequence's annotations as a track table.
    rows = []
    for frame in ET.parse(xml_path).getroot().iter("frame"):
        index = int(frame.get("num")) - 1 # UA-DETRAC counts frames from 1, video readers from 0
        for target in frame.iter("target"):
            box = target.find("box")
            left, top = float(box.get("left")), float(box.get("top"))
            rows.append({
                "frame": index,
                "t": index / FPS,
                "track_id": int(target.get("id")),
                "label": TYPE_TO_LABEL[target.find("attribute").get("vehicle_type")],
                "x1": left,
                "y1": top,
                "x2": left + float(box.get("width")),
                "y2": top + float(box.get("height")),
                "conf": 1.0
            })

    return pl.DataFrame(rows, schema=TRACK_SCHEMA).sort("frame", "track_id")

def load_ignored_regions(xml_path: Path) -> list[tuple[float, float, float, float]]:
    # The boxes (x1, y1, x2, y2) where UA-DETRAC did not label vehicles.
    regions = []
    for box in ET.parse(xml_path).getroot().find("ignored_region").iter("box"):
        left, top = float(box.get("left")), float(box.get("top"))
        regions.append((left, top, left + float(box.get("width")), top + float(box.get("height"))))
    return regions

