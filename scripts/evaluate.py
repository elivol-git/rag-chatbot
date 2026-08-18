"""Run the validation question set end to end and print a grounding report.

    python scripts/evaluate.py            # retrieval only (fast)
    python scripts/evaluate.py --answers  # also call the LLM for each question

Each case lists keywords expected in the retrieved context. The final case is
the grounding control: it must retrieve nothing above MIN_SCORE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Corpus text is UTF-8; Windows consoles often are not (cp1255, cp1252, ...).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import settings  # noqa: E402
from src.llm import chat  # noqa: E402
from src.prompts import build_prompt  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

CASES = [
    # Course material (Hebrew and English), from data/documents/michlala
    ("מהי אדריכלות מדברית?", ["מדבר", "אקלים", "קרינה"]),
    ("מה זה גג ירוק ומה היתרונות שלו?", ["גג", "ירוק", "בידוד"]),
    ("איך עובד איוורור טבעי בבניין?", ["איוורור", "אוויר", "טבעי"]),
    ("מה ההבדל בין בטון מזוין לבטון דרוך?", ["בטון", "פלדה"]),
    ("What characterises the architecture of Israel?", ["israel", "architecture"]),
    # General corpus (Wikipedia / Gutenberg)
    ("What are Vitruvius' three principles of good architecture?", ["firmitas", "utilitas", "venustas", "strength", "utility", "beauty"]),
    ("What defines Brutalist architecture?", ["brutalis", "concrete", "béton"]),
    ("Who founded the Bauhaus and what did it teach?", ["gropius", "bauhaus"]),
    ("How does a flying buttress work?", ["buttress", "thrust", "vault"]),
    ("What is the difference between Romanesque and Gothic arches?", ["pointed", "arch", "romanesque", "gothic"]),
    ("What is a passive house?", ["passive", "insulation", "energy"]),
    ("What did Louis Sullivan mean by form follows function?", ["sullivan", "form", "function"]),
    ("How do I configure a Kubernetes ingress controller?", []),  # grounding control
    ("איך מכינים חומוס?", []),  # grounding control, Hebrew
]


OFF_TOPIC = [
    "How do I configure a Kubernetes ingress controller?",
    "איך מכינים חומוס?",
    "What is the offside rule in football?",
    "מתי יוצא הטלפון החדש של סמסונג?",
    "Explain monetary policy and interest rates.",
]


def calibrate() -> int:
    """Report the gap between on-topic and off-topic top scores.

    MIN_SCORE has to sit between the two, and the right value depends on the
    embedding model, so it must be re-measured whenever EMBED_MODEL changes.
    """
    on = [(q, retrieve(q, top_k=1, min_score=0.0)) for q, kw in CASES if kw]
    off = [(q, retrieve(q, top_k=1, min_score=0.0)) for q in OFF_TOPIC]

    print("on-topic (want these above the floor):")
    on_scores = []
    for question, hits in on:
        score = hits[0].score if hits else 0.0
        on_scores.append(score)
        print(f"  {score:.3f}  {question}")

    print("\noff-topic (want these below the floor):")
    off_scores = []
    for question, hits in off:
        score = hits[0].score if hits else 0.0
        off_scores.append(score)
        print(f"  {score:.3f}  {question}")

    lowest_on, highest_off = min(on_scores), max(off_scores)
    print(f"\nlowest on-topic  {lowest_on:.3f}")
    print(f"highest off-topic {highest_off:.3f}")
    if lowest_on <= highest_off:
        print("NO SAFE FLOOR: the two ranges overlap, retrieval needs work first")
        return 1
    print(f"suggested MIN_SCORE = {(lowest_on + highest_off) / 2:.2f}  (current {settings.min_score})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also generate LLM answers")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="report the score spread of on-topic vs off-topic queries and exit",
    )
    args = parser.parse_args()

    if args.calibrate:
        return calibrate()

    print(f"top_k={settings.top_k}  min_score={settings.min_score}  model={settings.llm_model}\n")
    failures = 0

    for question, keywords in CASES:
        started = time.time()
        chunks = retrieve(question)
        elapsed = int((time.time() - started) * 1000)
        context = " ".join(c.text.lower() for c in chunks)

        if not keywords:  # control case: must refuse
            ok = not chunks
            detail = "refused (no chunks above floor)" if ok else (
                f"LEAKED {[f'{c.source}:{c.score:.2f}' for c in chunks]}"
            )
        else:
            hit = [k for k in keywords if k.lower() in context]
            ok = bool(hit)
            top = f"{chunks[0].source} {chunks[0].score:.3f}" if chunks else "no chunks"
            detail = f"top={top} matched={hit}"

        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question}\n        {detail}  ({elapsed} ms)")

        if args.answers and chunks:
            answer = chat(build_prompt(question, chunks))
            print(f"        -> {answer[:400]}")
        print()

    print(f"{len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
