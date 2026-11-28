"""Ask one question about one frame with several free Gemini models; print what each returns.

Run from the repo root. Shows which free models answer right now, how fast, and at what
cost. The log lines show every HTTP request and the status Google returned.
"""
import logging
from pathlib import Path

from clipqa.llm import ask_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

frame = Path("data/MOT17/train/MOT17-09-FRCNN/img1/000301.jpg").read_bytes()  # MOT17-09 at 10 s
question = "How many people can you see in this image? Answer with one number."
for model in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-4-hydrogen"]:
    try:
        print(f"\n{model}: {ask_model(question, [frame], model=model, thinking='low')}\n")
    except Exception as error:  # a quick survey: report the failure and move on to the next model
        print(f"\n{model}: FAILED with {type(error).__name__}: {str(error)[:200]}\n")