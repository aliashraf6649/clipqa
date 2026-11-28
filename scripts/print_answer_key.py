"""Print every eval question with its correct answer. Run from the repo root."""
from pathlib import Path

from clipqa.answer_key import ground_truth, load_questions, true_answer

for q in load_questions(Path("evals/questions.toml")):
    print(f"{q['id']:8} {q['lang']}  {true_answer(q, ground_truth(q['clip']))!s:>16}  {q['text']}")