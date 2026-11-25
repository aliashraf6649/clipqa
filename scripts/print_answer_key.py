"""Print every eval question with its correct answer. Run from the repo root."""
from pathlib import Path

from clipqa.answer_key import load_questions, true_answer
from clipqa.ground_truth import load_mot_gt
from clipqa.track_tools import visible_only

MOT_DIR = Path("data/MOT17/train")

for q in load_questions(Path("evals/questions.toml")):
    gt = visible_only(load_mot_gt(MOT_DIR / f"{q['clip']}-FRCNN"))
    print(f"{q['id']}  {q['lang']}  {true_answer(q, gt)!s:>6}  {q['text']}")