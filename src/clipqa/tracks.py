"""The track table: one row per object per frame.

The ground truth (clipqa.ground_truth) and the YOLO tracker both produce tables
with these columns, so the same question-answering code can run on either.
"""

"""The track table: one row per object per frame.

The ground truth (clipqa.ground_truth) and the YOLO tracker (clipqa.tracking) both
produce tables with these columns, so the same question-answering code can run on
either.
"""
import polars as pl

TRACK_SCHEMA = {
    "frame": pl.Int64,     # frame index in the clip, counting from 0
    "t": pl.Float64,       # seconds from the start of the clip: frame / fps
    "track_id": pl.Int64,  # an object keeps the same id in every frame it appears in
    "label": pl.String,    # COCO class name, as YOLO uses: person, bicycle, car, ...
    "x1": pl.Float64,      # top-left corner of the box, in pixels
    "y1": pl.Float64,
    "x2": pl.Float64,      # bottom-right corner of the box, in pixels
    "y2": pl.Float64,
    "conf": pl.Float64,    # detector confidence; 1.0 for ground truth
}
TRACK_COLUMNS = list(TRACK_SCHEMA)