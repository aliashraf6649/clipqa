# The eval answer key: each question's correct answer, computed from the ground truth.
import tomllib
from pathlib import Path
import polars as pl
from clipqa.track_tools import count_at, count_crossings, count_objects, first_seen

# The functions a question's `truth` can name

TOOLS = {
    "count_objects": count_objects,
    "count_at": count_at,
    "count_crossings": count_crossings,
    "first_seen": first_seen
}


def load_questions(path: Path) -> list[dict]:
    # Read the eval questions from a TOML file.
    with open(path, "rb") as f:
        return tomllib.load(f)["question"]

def true_answer(question: dict, gt: pl.DataFrame) -> int | float | None:
    # Compute the correct answer to a question from the ground truth.
    truth = question["truth"]
    if "value" in truth:
        return truth["value"]
    args = {key: value for key, value in truth.items() if key != "tool"}
    result = TOOLS[truth["tool"]](gt, **args)
    if question["answer_type"] == "yes_no":
        return "yes" if result > 0 else "no"
    return result

