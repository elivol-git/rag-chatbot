"""Import lecturer-authored architecture material into data/documents/michlala/.

Each accepted file is converted to a small .md file holding only its extracted
text. The originals stay where they are: the decks run to hundreds of megabytes
of images, and only text is ever embedded. Converting also leaves the Office
author/revision metadata behind.

The source folder mixes teaching material with graded work, exams and student
submissions. Only teaching material belongs in a knowledge base, so this script
is deliberately conservative:

  * only the architecture-related folders/files listed in CANDIDATES are read
  * anything whose name marks it as an exam, quiz or answer key is skipped
  * anything whose text contains an Israeli ID number, a student ID line or a
    "name of the lecturer" submission header is skipped as student work
  * files that yield too little text to be worth embedding are skipped

Nothing is deleted or modified at the source. Run with --dry-run first.

    python scripts/import_course_docs.py --dry-run
    python scripts/import_course_docs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import settings  # noqa: E402
from src.loaders import load_document  # noqa: E402

SOURCE_ROOT = Path(r"C:/Users/Daniel/Documents/Eli/Tanya_michlala")
TARGET = settings.documents_dir / "michlala"

# Folders and single files that hold architecture teaching material.
CANDIDATES = [
    "architecture",
    "aklim",
    "building",
    "2025/חומרי בניה",
    "Architecture of IL.docx",
    "Architecture_Il.pptx",
    "baroque.pptx",
    "brutalist.pptx",
    "DOV KARMI.pptx",
    "Yaakov Rechter.pptx",
    "Yaski.pptx",
    "environment.docx",
]

# Assessment material: keeping it out stops the bot from serving exam answers.
# Question sets (שאלון, מאגר שאלות, שאלות על) are excluded too - they are lists
# of questions without answers, so they add nothing a grounded bot can use and
# they leak upcoming exam items.
EXAM_WORDS = re.compile(
    r"exam|midterm|final|quiz|test|moed|מבחן|בוחן|מועד|answers?[_ -]?key|ptor"
    r"|שאלון|מאגר שאלות|שאלות על",
    re.IGNORECASE,
)

# Only unambiguous markers, applied to file text: a graded cover page or an
# answer key. Kept narrow so a teaching deck that merely poses questions stays.
EXAM_CONTENT = re.compile(
    r"מועד\s*[אב]\b|answer\s*key|מחוון|ציון\s*:|שם\s*התלמיד", re.IGNORECASE
)

# Student submissions: an Israeli ID number, a submission header, or a graded
# cover slide. Any of these means the file carries personal data.
ID_NUMBER = re.compile(r'(?:ת["\u05f4]?\s?ז|תעודת זהות|id\s*number)\D{0,12}\d{7,9}', re.IGNORECASE)
SUBMISSION_HEADER = re.compile(
    r"name of the lecturer|שם\s*(ה)?מרצה|submitted by|מגיש[הת]?\s*:", re.IGNORECASE
)

MIN_CHARS = 400


def candidate_files() -> list[Path]:
    files: list[Path] = []
    for entry in CANDIDATES:
        path = SOURCE_ROOT / entry
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            print(f"missing   {entry}")
    return files


def rejection_reason(path: Path, text: str | None) -> str | None:
    if path.name.startswith("~$"):
        return "temp file"
    if path.suffix.lower() not in {".docx", ".pptx", ".pdf", ".txt", ".md"}:
        return f"unsupported {path.suffix or 'no suffix'}"
    if EXAM_WORDS.search(path.name):
        return "assessment material"
    if text is None:
        return "no extractable text"
    if len(text) < MIN_CHARS:
        return f"too short ({len(text)} chars)"
    if ID_NUMBER.search(text):
        return "contains an ID number (student work)"
    if SUBMISSION_HEADER.search(text):
        return "student submission header"
    if EXAM_CONTENT.search(text[:600]):
        return "assessment material (by content)"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report without copying")
    args = parser.parse_args()

    if not SOURCE_ROOT.exists():
        print(f"Source folder not found: {SOURCE_ROOT}", file=sys.stderr)
        return 1

    accepted: list[tuple[Path, str]] = []
    rejected: list[tuple[Path, str]] = []

    for path in candidate_files():
        text = None
        if path.suffix.lower() in {".docx", ".pptx", ".pdf", ".txt", ".md"} and not path.name.startswith("~$"):
            try:
                document = load_document(path, path.parent)
                text = document.text if document else None
            except Exception as exc:  # noqa: BLE001 - a broken file is just skipped
                rejected.append((path, f"unreadable ({type(exc).__name__})"))
                continue

        reason = rejection_reason(path, text)
        if reason:
            rejected.append((path, reason))
        else:
            accepted.append((path, text or ""))

    print(f"\n{len(accepted)} accepted, {len(rejected)} skipped\n")
    for path, reason in rejected:
        print(f"  skip  {path.relative_to(SOURCE_ROOT).as_posix():55} {reason}")
    print()
    for path, text in accepted:
        print(f"  take  {path.relative_to(SOURCE_ROOT).as_posix():55} {len(text):,} chars")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    for path, text in accepted:
        relative = path.relative_to(SOURCE_ROOT)
        # Flatten into one folder, keeping the sub-folder in the name.
        # Keep the extension in the name: the same topic often exists as both a
        # .docx and a .pptx, and dropping it would silently collide.
        flat = "-".join(relative.parts).replace(" ", "_").replace(".", "_")
        target = TARGET / f"{flat}.md"
        header = (
            f"title: {path.stem}\n"
            f"source_url: {relative.as_posix()}\n"
            f"license: Course material, Tanya Volowelsky\n"
            f"---\n\n"
        )
        target.write_text(header + text.strip() + "\n", encoding="utf-8")

    written = sum(p.stat().st_size for p in TARGET.glob("*.md"))
    print(f"\nwrote {len(accepted)} text file(s) into {TARGET} ({written / 1_048_576:.1f} MB)")
    print("next: python -m src.ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
