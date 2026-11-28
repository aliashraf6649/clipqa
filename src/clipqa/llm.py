"""The one place ClipQA talks to an LLM. To switch providers, change this file only."""
import base64
import time
from dataclasses import dataclass
from functools import cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Measured on 2026-11-28: on the free tier, gemini-3.5-flash-lite answered in about 4 s,
# while gemini-3.6-flash, gemini-3.8-flash and gemini-4-hydrogen were overloaded
# (503s and timeouts). Bigger models can still be tried per call with `model=`.
MODEL = "gemini-3.5-flash-lite"


@dataclass
class Reply:
    """A model's answer, with what it cost."""
    text: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    seconds: float


TIMEOUT_MS = 90_000  # give up on one request after 90 s (the SDK's default is to wait forever)
ATTEMPTS = 2         # the first try plus 1 retry on 408, 429 and 5xx errors (the default is 5)


@cache
def _client() -> genai.Client:
    """Create the Gemini client once, on first use."""
    load_dotenv()  # puts GEMINI_API_KEY from the repo's .env into the environment
    return genai.Client(http_options=types.HttpOptions(
        timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=ATTEMPTS)))


def ask_model(prompt: str, images: list[bytes] | None = None, model: str = MODEL,
              thinking: str | None = None) -> Reply:
    """Send a text prompt plus optional JPEG images; return the reply and its cost.

    `thinking` sets how much the model reasons before answering ("low", "medium",
    "high"); None keeps the model's default.
    """
    parts = [
        {"type": "image", "data": base64.b64encode(image).decode("utf-8"), "mime_type": "image/jpeg"}
        for image in images or []
    ]
    options = {"generation_config": {"thinking_level": thinking}} if thinking else {}
    start = time.perf_counter()
    interaction = _client().interactions.create(
        model=model, input=parts + [{"type": "text", "text": prompt}], **options
    )
    usage = interaction.usage
    return Reply(
        text=interaction.output_text,
        input_tokens=usage.total_input_tokens or 0,
        output_tokens=usage.total_output_tokens or 0,
        thinking_tokens=usage.total_thought_tokens or 0,
        seconds=time.perf_counter() - start,
    )