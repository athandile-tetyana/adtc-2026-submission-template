"""
pdf_to_chunks.py — ADTC 2026 Agriculture RAG: corpus extraction + chunking

WHAT THIS DOES (plain English version, for explaining to your brother):
  You point it at a folder of agriculture PDFs. It reads each one, pulls out
  the raw text, and cuts that text into bite-sized "chunks" — small enough
  that the model can actually read one at inference time, but big enough
  that each chunk still makes sense on its own (a whole section about maize
  fertilizer rates, not half a sentence). It saves those chunks as JSON so
  the next script (embedding + FAISS index) can turn them into a searchable
  vector store.

WHY CHUNK ON HEADINGS INSTEAD OF FIXED TOKEN WINDOWS:
  Fixed-size windows (e.g. "every 500 characters") cut sentences and even
  crop-specific sections in half — a chunk might end mid-way through the
  tomato fertilizer rate and start the next chunk mid-way through pest
  control for a different crop. Since these ARC/NDA guides already have
  clean section headers per crop, we chunk on those boundaries first, and
  only fall back to size-based splitting if a section is too long.

USAGE:
  python pdf_to_chunks.py --input corpus/raw --output corpus/chunks.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


# --- Config -----------------------------------------------------------------

# Max characters per chunk. Kept small on purpose: at inference time these
# chunks get stuffed into the model's context window alongside the user's
# question and the model's own reasoning. On an 8GB laptop with a 512-2048
# token context, you can't afford 3000-character chunks eating the budget.
MAX_CHUNK_CHARS = 1200

# Minimum chunk size — anything shorter than this gets merged into the
# next chunk instead of being retrieved on its own (a 2-line orphan chunk
# with no context is useless to the model).
MIN_CHUNK_CHARS = 200

# Headings in these government/ARC guides are typically ALL CAPS lines,
# or short lines (under ~60 chars) ending without a period, often followed
# by a colon. This regex is intentionally loose — better to over-detect
# section breaks than under-detect them, since merging is cheap (see
# MIN_CHUNK_CHARS above) but mis-splitting mid-sentence is not.
HEADING_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s/&\-]{4,60}|(?:\d+\.\s?)?[A-Z][a-zA-Z\s/&\-]{4,50}:)\s*$"
)


# --- Extraction ---------------------------------------------------------------

def extract_text_by_page(pdf_path: Path) -> list[str]:
    """
    Pull raw text out of a PDF, one string per page.

    Why per-page and not one giant blob: we tag each chunk with its source
    page number later, so when the model cites "planting dates" you can
    trace it back to page 4 of the maize guide, not just "somewhere in
    this 40-page document."
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Collapse the multiple-spaces/broken-line mess that PDF text
        # extraction often produces from multi-column layouts.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages


# --- Chunking -----------------------------------------------------------------

def split_into_sections(page_text: str) -> list[str]:
    """
    Break one page's text into sections using heading-like lines as the
    split points. If no headings are detected, the whole page comes back
    as a single section (handled by the length-based fallback next).
    """
    lines = page_text.split("\n")
    sections = []
    current: list[str] = []

    for line in lines:
        if HEADING_PATTERN.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if s]


def _hard_split(text: str) -> list[str]:
    """
    Last-resort splitter for text with no paragraph breaks to split on
    (continuous single-newline text, common when a PDF's layout doesn't
    produce blank lines during extraction). Splits on sentence boundaries
    instead, so we still avoid cutting mid-sentence even without paragraph
    structure to lean on.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 <= MAX_CHUNK_CHARS:
            buf = f"{buf} {sentence}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sentence
    if buf:
        chunks.append(buf)
    return chunks


def enforce_size_limits(sections: list[str]) -> list[str]:
    """
    Two cleanup passes over the heading-based sections:
      1. Anything over MAX_CHUNK_CHARS gets split further, on paragraph
         breaks where possible, so we never blow the context budget.
      2. Anything under MIN_CHUNK_CHARS gets merged into the chunk before
         it, so retrieval never returns a near-empty fragment.
    """
    # Pass 1: split oversized sections on paragraph boundaries
    sized = []
    for section in sections:
        if len(section) <= MAX_CHUNK_CHARS:
            sized.append(section)
            continue
        paragraphs = section.split("\n\n")
        buf = ""
        for para in paragraphs:
            # A single "paragraph" can itself exceed MAX_CHUNK_CHARS when the
            # source PDF has no blank-line breaks at all (continuous text,
            # one newline per line). Without this guard, such a paragraph
            # sails through untouched and never gets chunked - this is what
            # produced the 5000-char blob from Brochure Fruit vegetables.pdf.
            # Fall back to a hard split on sentence boundaries when a single
            # paragraph alone is too big.
            if len(para) > MAX_CHUNK_CHARS:
                if buf:
                    sized.append(buf)
                    buf = ""
                sized.extend(_hard_split(para))
                continue
            if len(buf) + len(para) + 2 <= MAX_CHUNK_CHARS:
                buf = f"{buf}\n\n{para}".strip()
            else:
                if buf:
                    sized.append(buf)
                buf = para
        if buf:
            sized.append(buf)

    # Pass 2: merge undersized chunks forward into their neighbor
    merged = []
    for chunk in sized:
        if merged and len(chunk) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)

    return merged


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """
    Full pipeline for one PDF: extract -> split into headed sections per
    page -> enforce size limits -> attach metadata each chunk needs for
    citation and filtering later (source file, page number, chunk index).
    """
    pages = extract_text_by_page(pdf_path)
    chunks = []
    chunk_idx = 0

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        sections = split_into_sections(page_text)
        sized_sections = enforce_size_limits(sections)
        for section in sized_sections:
            if len(section.strip()) < 30:  # skip near-empty noise
                continue
            chunks.append({
                "id": f"{pdf_path.stem}_{chunk_idx:04d}",
                "source": pdf_path.name,
                "page": page_num,
                "text": section.strip(),
            })
            chunk_idx += 1

    return chunks


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder of raw PDFs")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        chunks = chunk_pdf(pdf_path)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
