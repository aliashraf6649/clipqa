"""The track table: one row per object per frame.

The ground truth (clipqa.ground_truth) and the YOLO tracker both produce tables
with these columns, so the same question-answering code can run on either.
"""

TRACK_COLUMNS = [
    "frame",     # frame index in the clip, counting from 0
    "t",         # seconds from the start of the clip: frame / fps
    "track_id",  # an object keeps the same id in every frame it appears in
    "label",     # COCO class name, as YOLO uses: person, bicycle, car, ...
    "x1", "y1",  # top-left corner of the box, in pixels
    "x2", "y2",  # bottom-right corner of the box, in pixels
    "conf",      # detector confidence; 1.0 for ground truth
]