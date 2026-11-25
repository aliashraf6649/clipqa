"""Load MOTChallenge ground truth as a ClipQA track table."""
import configparser
from pathlib import Path

import polars as pl

from clipqa.tracks import TRACK_COLUMNS

# The columns of MOT17's gt/gt.txt, which has no header row. Boxes are read as
# floats to match YOLO's boxes.
GT_TXT_SCHEMA = {
    "mot_frame": pl.Int64,
    "track_id": pl.Int64,
    "left": pl.Float64,
    "top": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "consider": pl.Int64,
    "mot_class": pl.Int64,
    "visibility": pl.Float64,
}

# MOT17 classes we keep, renamed to YOLO's COCO labels. Everything else
# (other vehicles, distractors, occluders, reflections) is dropped.
MOT_CLASS_TO_LABEL = {
    1: "person",  # pedestrian
    2: "person",  # person on a vehicle, e.g. a cyclist
    3: "car",
    4: "bicycle",
    5: "motorcycle",
    7: "person",  # static person: sitting or lying down
}


def load_mot_gt(seq_dir: Path) -> pl.DataFrame:
    """Read one MOT17 sequence's ground truth as a track table.

    Adds two columns only the ground truth has: visibility (0-1, how much of the
    object is not hidden) and mot_class (the original MOT17 class id).
    """
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    fps = int(info["Sequence"]["frameRate"])

    gt = pl.read_csv(seq_dir / "gt" / "gt.txt", has_header=False, schema=GT_TXT_SCHEMA)
    return (
        gt.filter(pl.col("mot_class").is_in(list(MOT_CLASS_TO_LABEL)))
        .with_columns(
            frame=pl.col("mot_frame") - 1,  # MOT counts frames from 1, video readers from 0
            label=pl.col("mot_class").replace_strict(MOT_CLASS_TO_LABEL),
            x1=pl.col("left"),
            y1=pl.col("top"),
            x2=pl.col("left") + pl.col("width"),
            y2=pl.col("top") + pl.col("height"),
            conf=pl.lit(1.0),
        )
        .with_columns(t=pl.col("frame") / fps)
        .select(*TRACK_COLUMNS, "visibility", "mot_class")
        .sort("frame", "track_id")
    )