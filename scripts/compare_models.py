"""Compare generator models on identical retrieved context.

Retrieval runs once per question and the same passages are handed to every
model, so differences in the output are the model's, not the retriever's.

    python scripts/compare_models.py aya-expanse:8b gemma3:4b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests  # noqa: E402

from src import llm  # noqa: E402
from src.config import settings  # noqa: E402
from src.prompts import build_prompt  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

QUESTIONS = [
    "מה זה גג ירוק ומה היתרונות שלו?",
    "מה ההבדל בין בטון מזוין לבטון דרוך?",
    "What defines Brutalist architecture?",
]


def generate(model: str, messages: list[dict]) -> tuple[str, float, float]:
    """Return (answer, seconds_to_first_token, total_seconds) for one model."""
    started = time.time()
    first = None
    text = ""

    with requests.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": llm.KEEP_ALIVE,
            "options": {"temperature": 0.2, "num_predict": settings.max_answer_tokens},
        },
        timeout=llm.TIMEOUT,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            import json

            payload = json.loads(line)
            fragment = payload.get("message", {}).get("content", "")
            if fragment and first is None:
                first = time.time() - started
            text += fragment
            if payload.get("done"):
                break

    return text, first or 0.0, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Ollama model names to compare")
    parser.add_argument("--chars", type=int, default=700, help="answer preview length")
    args = parser.parse_args()

    for question in QUESTIONS:
        chunks = retrieve(question)
        print("=" * 78)
        print(question)
        print(f"context: {len(chunks)} passages, "
              f"{sum(len(c.text) for c in chunks):,} chars, "
              f"top score {chunks[0].score:.3f}" if chunks else "context: none")
        messages = build_prompt(question, chunks)

        for model in args.models:
            print(f"\n--- {model} ---")
            try:
                answer, first, total = generate(model, messages)
            except requests.RequestException as exc:
                print(f"failed: {exc}")
                continue
            rate = len(answer) / total if total else 0
            print(f"first token {first:.1f}s | total {total:.1f}s | {rate:.0f} chars/s")
            print(answer[: args.chars])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
