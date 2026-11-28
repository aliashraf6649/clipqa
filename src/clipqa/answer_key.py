"""The eval answer key: each question's correct answer, computed from the ground truth."""
import tomllib
from functools import cache
from pathlib import Path

import polars as pl

from clipqa.detrac import load_detrac_gt
from clipqa.ground_truth import load_mot_gt
from clipqa.track_tools import (
    count_at,
    count_crossings,
    count_objects,
    drop_short_tracks,
    first_seen,
    visible_only,
)

MOT_DIR = Path("data/MOT17/train")
DETRAC_XML_DIR = Path("data/DETRAC/DETRAC-Test-Annotations-XML")

# The functions a question's `truth` can name.
TOOLS = {
    "count_objects": count_objects,
    "count_at": count_at,
    "count_crossings": count_crossings,
    "first_seen": first_seen,
}


def load_questions(path: Path) -> list[dict]:
    """Read the eval questions from a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)["question"]


@cache
def ground_truth(clip: str) -> pl.DataFrame:
    """One clip's ground truth, keeping only what a viewer could really see."""
    if clip.startswith("MVI_"):  # UA-DETRAC has no visibility labels: only the 0.5 s rule
        return drop_short_tracks(load_detrac_gt(DETRAC_XML_DIR / f"{clip}.xml"))
    return visible_only(load_mot_gt(MOT_DIR / f"{clip}-FRCNN"))


def tool_answer(question: dict, tracks: pl.DataFrame) -> int | float | str | None:
    """Answer a question by running its tool on a track table."""
    truth = question["truth"]
    args = {key: value for key, value in truth.items() if key != "tool"}
    result = TOOLS[truth["tool"]](tracks, **args)
    if question["answer_type"] == "yes_no":
        return "yes" if result > 0 else "no"
    return result


def true_answer(question: dict, gt: pl.DataFrame) -> int | float | str | None:
    """Compute one question's correct answer from its clip's visible ground truth."""
    if "value" in question["truth"]:  # labelled by hand, e.g. a colour
        return question["truth"]["value"]
    return tool_answer(question, gt)


def is_correct(question: dict, predicted, truth) -> bool:
    """Whether a predicted answer counts as right.

    Counts may be off by 10% (at least 1), times by 1 second; anything else must match.
    """
    kind = question["answer_type"]
    if kind == "count":
        return abs(predicted - truth) <= max(1, 0.1 * truth)
    if kind == "seconds":
        if predicted is None or truth is None:
            return predicted is None and truth is None
        return abs(predicted - truth) <= 1.0
    return str(predicted).lower() == str(truth).lower()