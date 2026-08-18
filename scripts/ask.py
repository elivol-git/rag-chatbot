"""Ask the knowledge base one question from the command line.

Uses the same retrieve -> build_prompt -> chat path as the Flask API, so it is
a quick way to check answer quality without starting the server.

    python scripts/ask.py "מהי אדריכלות מדברית?"
    python scripts/ask.py --top-k 6 "What is reinforced concrete?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.llm import chat_stream  # noqa: E402
from src.prompts import build_prompt, detect_language, refusal_for  # noqa: E402
from src.retrieval import retrieve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--source", default=None, help="restrict to one document")
    args = parser.parse_args()

    started = time.time()
    chunks = retrieve(args.question, top_k=args.top_k, source_filter=args.source)
    retrieval_ms = int((time.time() - started) * 1000)

    print(f"language: {detect_language(args.question)}   retrieval: {retrieval_ms} ms")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}] {chunk.score:.3f}  {chunk.source}")
    print()

    if not chunks:
        print(refusal_for(args.question))
        return 0

    llm_started = time.time()
    for fragment in chat_stream(build_prompt(args.question, chunks)):
        print(fragment, end="", flush=True)
    print(f"\n\n({int((time.time() - llm_started) * 1000)} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
