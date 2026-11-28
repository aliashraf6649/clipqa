"""Score YOLO's tracks: answer each question with its tool, compare with the answer key.

Run from the repo root. Hand-labelled questions are skipped: they need the vision model.
"""
from pathlib import Path

import polars as pl

from clipqa.answer_key import is_correct, load_questions, tool_answer, true_answer
from clipqa.ground_truth import load_mot_gt
from clipqa.track_tools import drop_short_tracks, visible_only

MOT_DIR = Path("data/MOT17/train")
TRACKS_DIR = Path("data/tracks/yolo26m")

right_all, right_kept, total = 0, 0, 0
print(f"{'id':6} {'tool':16} {'truth':>6} {'all tracks':>11} {'>= 0.5 s':>10}")
for q in load_questions(Path("evals/questions.toml")):
    if "tool" not in q["truth"]:
        continue
    truth = true_answer(q, visible_only(load_mot_gt(MOT_DIR / f"{q['clip']}-FRCNN")))
    tracks = pl.read_parquet(TRACKS_DIR / f"{q['clip']}.parquet")
    answer_all = tool_answer(q, tracks)
    answer_kept = tool_answer(q, drop_short_tracks(tracks))
    ok_all, ok_kept = is_correct(q, answer_all, truth), is_correct(q, answer_kept, truth)
    right_all += ok_all
    right_kept += ok_kept
    total += 1
    print(f"{q['id']:6} {q['truth']['tool']:16} {truth!s:>6} {answer_all!s:>9} {'✓' if ok_all else '✗'}"
          f" {answer_kept!s:>8} {'✓' if ok_kept else '✗'}")
print(f"\nCorrect: {right_all}/{total} using all tracks, {right_kept}/{total} using tracks of at least 0.5 s")