"""Questions you can ask a track table.

These functions answer questions from any track table (see clipqa.tracks). Run on
the ground truth, they produce the eval answer key; run on YOLO's tracks, they
become the agent's tools.
"""

import polars as pl

def visible_only(gt: pl.DataFrame, min_visibility: float = 0.5, min_frames: int = 15) -> pl.DataFrame:
    """Keep only the ground truth a viewer could really see.

    Drops rows where the object is less than `min_visibility` visible, then drops
    objects left with fewer than `min_frames` frames.
    """
    return (
        gt.filter(pl.col("visibility") >= min_visibility)
        .filter(pl.len().over("track_id") >= min_frames)
    )

def count_objects (tracks: pl.DataFrame, label: str, t_from: float = 0.0, t_to: float | None = None) -> int:
    # How many different objects with this label appear between t_from and t_to.(seconds)
    rows = tracks.filter(pl.col("label") == label, pl.col("t") >= t_from)
    if t_to is not None:
        rows = rows.filter(pl.col("t") <= t_to)
    return rows["track_id"].n_unique()

def count_at(tracks: pl.DataFrame, label: str, t: float) -> int:
    # How many objects with this label are in the frame closest to time t
    frame = tracks.sort((pl.col("t") - t).abs())["frame"][0]
    return tracks.filter(pl.col("label") == label, pl.col("frame") ==frame).height

def count_crossings(tracks: pl.DataFrame, label: str, x: float, direction: str) -> int:
    """How many objects with this label cross the vertical line at x.

    Compares the side of the line where each object is first seen with the side
    where it is last seen. direction is "left_to_right" or "right_to_left".
    """
    ends = (
        tracks.filter(pl.col("label") == label)
        .sort("frame")
        .with_columns(cx=(pl.col("x1") + pl.col("x2")) / 2)
        .group_by("track_id")
        .agg(
            start_x = pl.col("cx").first(),
            end_x = pl.col("cx").last(),
        )
    )
    if direction == "left_to_right":
        crossed = ends.filter(pl.col("start_x") < x, pl.col("end_x") >= x)
    elif direction == "right_to_left":
        crossed = ends.filter(pl.col("start_x") >= x, pl.col("end_x") < x)
    else:
        raise ValueError(f"direction must be left_to_right or right_to_left, not {direction!r}")
    return crossed.height

def first_seen(tracks: pl.DataFrame, label: str) -> float | None:
    """When an object with this label is first visible, in seconds; None if never."""
    rows = tracks.filter(pl.col("label") == label)
    return rows["t"].min() if rows.height > 0 else None