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

from src.config import settings  # noqa: E402
from src.llm import chat  # noqa: E402
from src.prompts import build_prompt  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

CASES = [
    ("What are Vitruvius' three principles of good architecture?", ["firmitas", "utilitas", "venustas", "strength", "utility", "beauty"]),
    ("What defines Brutalist architecture?", ["brutalis", "concrete", "béton"]),
    ("Who founded the Bauhaus and what did it teach?", ["gropius", "bauhaus"]),
    ("How does a flying buttress work?", ["buttress", "thrust", "vault"]),
    ("What is the difference between Romanesque and Gothic arches?", ["pointed", "arch", "romanesque", "gothic"]),
    ("What is a passive house?", ["passive", "insulation", "energy"]),
    ("What did Louis Sullivan mean by form follows function?", ["sullivan", "form", "function"]),
    ("How do I configure a Kubernetes ingress controller?", []),  # grounding control
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also generate LLM answers")
    args = parser.parse_args()

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
