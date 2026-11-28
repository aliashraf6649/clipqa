# Track every clip with YOLO and save the track tables. Run from the repo root.

import time
from pathlib import Path

from clipqa.tracking import track_clip

MODEL = "yolo26m.pt"
CLIPS = ["MOT17-02", "MOT17-04", "MOT17-09"]
OUT_DIR = Path("data/tracks") / Path(MODEL).stem

OUT_DIR.mkdir(parents=True, exist_ok=True)
for clip in CLIPS:
    start = time.perf_counter()
    tracks = track_clip(Path("data/clips") / f"{clip}.mp4", MODEL)
    tracks.write_parquet(OUT_DIR / f"{clip}.parquet")
    seconds = time.perf_counter() - start
    print(f"{clip}: {tracks.height} boxes, {tracks['track_id'].n_unique()} track IDs, {seconds:.0f} s")